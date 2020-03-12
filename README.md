# Deep Learning Framework Inference Speed Benchmark - CPU only, Single thread

## Build Instructions
* You will need to have OpenCV installed on your system
* Navigate to 3rd_party_libs and run the shell scripts (note the libraries are built for target arch `broadwell`)
* Use the command line arguments to specify which framework to use for inference - `./inference_benchmarks --help` to show all options
* Place model files in `./models`

* `export LD_LIBRARY_PATH=/path/to/3rd_paty_libs/mxnet`

## Results - Single Process
Inference performed on 112x112 pre-aligned face chips (200 images), Single thread mode, using ResNet 100

`gcc (Ubuntu 9.2.1-17ubuntu1~18.04.1) 9.2.1 20191102`

`Ubuntu 18.04.3 LTS`

### `Intel(R) Core(TM) i5-7500T CPU @ 2.70GHz, 16GB ram`

**MXNET**: 302.728 ms per image

**NCNN**: 333.06 ms per image

**TVM**: 321.799 ms per image

**Tensorflow**: 354.973 ms per image

**Onnxruntime**: 274.19 ms per image

**OpenVINO**: Unable to get number of threads down to 1


### `Dual Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz, 128 GB RAM`

**NCNN**: 385.912 ms per image

**MXNET**: 352.022 ms per image

**TVM**: 408.418 ms per image

**Onnxruntime**: 312.286 ms per image


