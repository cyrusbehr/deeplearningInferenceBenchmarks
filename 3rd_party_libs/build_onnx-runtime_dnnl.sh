test -e onnxruntime || git clone --recursive https://github.com/Microsoft/onnxruntime
cd onnxruntime
git checkout tags/v1.6.0 
./build.sh --use_dnnl --parallel --build_shared_lib --config Release

