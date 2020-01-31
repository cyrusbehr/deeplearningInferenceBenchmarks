# Deep Learning Framework Inference Speed Benchmark - CPU only, Single thread

## Build Instructions
* You will need to have OpenCV installed on your system
* Navigate to 3rd_party_libs and run the shell scripts (note the libraries are built for target arch `broadwell`)
* Use the command line arguments to specify which framework to use for inference - `./inference_benchmarks --help` to show all options
* To run inference with tensorflow, run `cmake -DUSE_TENSORFLOW=ON`. Note, this will disable `mxnet` as there are conflicts between the two.
* Place model files in `./models`

* `export LD_LIBRARY_PATH=/path/to/3rd_paty_libs/mxnet`

## Results
Inference performed on 112x112 pre-aligned face chips (200 images), Single thread mode, using ResNet 100


Benchmark Machine Specs: 

`Intel(R) Core(TM) i5-7500T CPU @ 2.70GHz, 16GB ram` (4 cores)

`gcc (Ubuntu 9.2.1-17ubuntu1~18.04.1) 9.2.1 20191102`

`Ubuntu 18.04.3 LTS`


### Single process, single core, single thread

**MXNET**: 302.728 ms per image

**NCNN**: 333.06 ms per image

**TVM**: 321.799 ms per image

**Tensorflow**: 354.973 ms per image

**Onnxruntime**: 274.19 ms per image

**OpenVINO**: Unable to get number of threads down to 1

### 1 process per core (4 simultaneous), single thread (using `taskset --cpu-list`)

**MXNET**: 366.66 ms per image

**NCNN**: 573.36 ms per image

**TVM**: N/A (Was unable to get the 4 processes to run on seperate cores)

**Tensorflow**: 437.69 ms per image

**Onnxruntime**: 312.3775 ms per image


