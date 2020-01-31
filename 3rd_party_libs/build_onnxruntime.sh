test -e onnxruntime-linux-x64-1.1.0.tgz || wget https://github.com/microsoft/onnxruntime/releases/download/v1.1.0/onnxruntime-linux-x64-1.1.0.tgz
tar -xzvf ./onnxruntime-linux-x64-1.1.0.tgz
mv onnxruntime-linux-x64-1.1.0 onnxruntime