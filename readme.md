# Deep Learning Inference Framework Speed Benchmarks - CPU

## About 
This project benchmarks the performance of various popular deep learning inference frameworks on x86_64 CPU.
The intention is to determine which has the lowest latency, and how number of threads used for inference impacts performance.
Thead dependency can be used to optimize CPU throughput (ex. run multiple instances in parallel each only using a reduced number of threads).

## Test Specs
The benchmarks are performed on `Dual Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz`, 20 cores total, 128 Gb of ram.
The model used for inference is a ResNet100. The model input is a 112x112 rgb image. 
The time taken to read the image into memory and convert to rgb format is *not* included in the inference time.
The time taken to convert the rgb image buffer into the framework-specific expected network input format *is* included in the inference time. 
First inference time is discarded to ensure all network weights and params have been loaded. Memory usage refers to maximum resident set size.
The number of inferences used to generate the average time is on the order of 1000. 
*Unrestricted threads* in tables below means allowing the inference framework to choose how many threads it wants to use on it's own.  

## Usage
`./deeplearningInferenceBenchmarks /path/to/model/directory number_of_threads (optional)`

## Results

### ncnn
- Github [link](https://github.com/Tencent/ncnn)
- git release hash: `5e4ea0b`
- CMake build arguments: `-D NCNN_BUILD_EXAMPLES=OFF -D NCNN_VULKAN=OFF -D NCNN_AVX2=ON -D CMAKE_BUILD_TYPE=Release` 

| Number of threads 	| Average inference speed 	| Memory usage 	|
|-------------------	|-------------------------	|--------------	|
| 1                 	| 332 ms                  	| 1.72 Gb      	|
| 2                 	| 180 ms                  	| 1.72 Gb      	|
| 3                 	| 132 ms                  	| 1.72 Gb      	|
| 4                 	| 100 ms                  	| 1.72 Gb      	|
| 5                 	| 95 ms                   	| 1.72 Gb      	|
| 6                 	| 83 ms                   	| 1.72 Gb      	|
| 7                 	| 80 ms                   	| 1.72 Gb      	|
| 8                 	| 67 ms                   	| 1.72 Gb      	|
| 9                 	| 67 ms                   	| 1.72 Gb      	|
| 10                	| 68 ms                   	| 1.72 Gb      	|
| Unrestricted (20) 	| 66 ms                   	| 1.72 Gb      	|