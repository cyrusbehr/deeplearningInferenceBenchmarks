test -e mxnet || git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet

cd mxnet
mkdir build
cd build

cmake -DUSE_CPP_PACKAGE=1 -DUSE_CUDA=0 -DUSE_MKL_IF_AVAILABLE=1 -DUSE_OPENCV=0 -DUSE_LAPACK=0 -DUSE_OPENMP=0 \
-DMKL_INCLUDE_DIR=/opt/intel/compilers_and_libraries/linux/mkl/include -DMKL_RT_LIBRARY=/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_rt.so ..

if [ "$(uname)" == "Darwin" ]; then
    sysctl -n hw.physicalcpu | xargs -I % make -j%
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    nproc | xargs -I % make -j%
fi