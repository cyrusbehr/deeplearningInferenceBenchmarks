# Download ncnn
test -e incubator-mxnet || git clone --recursive https://github.com/apache/incubator-mxnet.git

cd incubator-mxnet
git reset --hard 2fc0706 # 1.8.0.rc3
git submodule update --init --recursive

mkdir build_amd64
cd build_amd64

cmake -DUSE_CPP_PACKAGE=1 -DBUILD_CPP_EXAMPLES=OFF -DUSE_CUDA=0 -DUSE_MKL_IF_AVAILABLE=1 -DUSE_BLAS=mkl -DUSE_OPENCV=0 -DUSE_LAPACK=0 -DUSE_OPENMP=1 \
 -DMKL_INCLUDE_DIR=/opt/intel/oneapi/mkl/latest/include -DMKL_RT_LIBRARY=/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_rt.so -DCMAKE_BUILD_TYPE=Release ..

nproc | xargs -I % make -j%

make DESTDIR=./packaged install

