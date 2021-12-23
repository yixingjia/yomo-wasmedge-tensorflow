package main

import "C"
import (
	"crypto/sha1"
	"fmt"
	"log"
	"os"
	"sync/atomic"

	"github.com/second-state/WasmEdge-go/wasmedge"
	"github.com/yomorun/yomo"
)

var (
	counter uint64
)

const ImageDataKey = 0x10
type VMContext struct {
	vm  *wasmedge.VM
	vmConf   *wasmedge.Configure
	tfImportObj *wasmedge.ImportObject
	tfLiteImportObj *wasmedge.ImportObject
	imgImportObj *wasmedge.ImportObject
}

func main() {
	// Connect to Zipper service
	sfn := yomo.NewStreamFunction("image-recognition", yomo.WithZipperAddr("localhost:9900"))
	defer sfn.Close()

	// set only monitoring data
	sfn.SetObserveDataID(ImageDataKey)

	// set handler
	err := sfn.SetHandler(Handler)
	if err != nil {
		log.Print("❌ Failed to set the handler: ", err)
		os.Exit(1)
	}

	// start
	err = sfn.Connect()
	if err != nil {
		log.Print("❌ Connect to zipper failure: ", err)
		os.Exit(1)
	}

	select {}
}

// Handler process the data in the stream
func Handler(img []byte) (byte, []byte) {
	// Initialize WasmEdge's VM
	vmContext, _:= initVM()
	defer func(){
		vmContext.imgImportObj.Release()
		vmContext.tfImportObj.Release()
		vmContext.tfLiteImportObj.Release()
		vmContext.vm.Release()
		vmContext.vmConf.Release()
	}()

	// recognize the image
	res, err := vmContext.vm.ExecuteBindgen("infer", wasmedge.Bindgen_return_array, img)
	if err == nil {
		fmt.Println("GO: Run bindgen -- infer:", string(res.([]byte)))
	} else {
		fmt.Println("GO: Run bindgen -- infer FAILED")
	}

	// print logs
	hash := genSha1(img)
	log.Printf("✅ received image-%d hash %v, img_size=%d \n", atomic.AddUint64(&counter, 1), hash, len(img))

	return 0x11, nil
}

// genSha1 generate the hash value of the image
func genSha1(buf []byte) string {
	h := sha1.New()
	h.Write(buf)
	return fmt.Sprintf("%x", h.Sum(nil))
}

// initVM initialize WasmEdge's VM
func initVM() (*VMContext, error) {
	wasmedge.SetLogErrorLevel()
	/// Set Tensorflow not to print debug info
	err := os.Setenv("TF_CPP_MIN_LOG_LEVEL", "3")
	if err != nil {
		return nil,err
	}
	err = os.Setenv("TF_CPP_MIN_VLOG_LEVEL", "3")
	if err != nil {
		return nil,err
	}

	/// Create configure
	vmContext := &VMContext{}
	vmContext.vmConf = wasmedge.NewConfigure(wasmedge.WASI)

	/// Create VM with configure
	vmContext.vm = wasmedge.NewVMWithConfig(vmContext.vmConf)

	/// Init WASI
	var wasi = vmContext.vm.GetImportObject(wasmedge.WASI)
	wasi.InitWasi(
		os.Args[1:],     /// The args
		os.Environ(),    /// The envs
		[]string{".:."}, /// The mapping directories
	)

	/// Register WasmEdge-tensorflow and WasmEdge-image
	var tfobj = wasmedge.NewTensorflowImportObject()
	var tfliteobj = wasmedge.NewTensorflowLiteImportObject()
	err = vmContext.vm.RegisterImport(tfobj)
	if err != nil {
		return nil, err
	}
	err = vmContext.vm.RegisterImport(tfliteobj)
	if err != nil {
		return nil, err
	}
	var imgobj = wasmedge.NewImageImportObject()
	err = vmContext.vm.RegisterImport(imgobj)
	if err != nil {
		return nil, err
	}

	/// Instantiate wasm
	err = vmContext.vm.LoadWasmFile("rust_mobilenet_food_lib_bg.so")
	if err != nil {
		return nil, err
	}
	err = vmContext.vm.Validate()
	if err != nil {
		return nil, err
	}
	err = vmContext.vm.Instantiate()
	if err != nil {
		return nil, err
	}

	return vmContext,nil
}
