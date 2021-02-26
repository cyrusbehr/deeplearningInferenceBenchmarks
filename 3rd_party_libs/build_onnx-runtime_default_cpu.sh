test -e onnxruntime-linux-x64-1.6.0.tgz || wget https://github.com/microsoft/onnxruntime/releases/download/v1.6.0/onnxruntime-linux-x64-1.6.0.tgz
test -e onnxruntime-linux-x64-1.6.0 && rm -r onnxruntime-linux-x64-1.6.0
tar -xzvf onnxruntime-linux-x64-1.6.0.tgz