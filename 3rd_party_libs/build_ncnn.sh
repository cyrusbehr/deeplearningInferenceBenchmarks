# Download ncnn
test -e ncnn || git clone --recursive https://github.com/Tencent/ncnn.git

cd ncnn
git reset --hard 5e4ea0b # Jan 24 release

mkdir build_amd64
cd build_amd64

cmake -D NCNN_BUILD_EXAMPLES=OFF -D NCNN_VULKAN=OFF -D NCNN_AVX2=ON -D CMAKE_BUILD_TYPE=Release ..

nproc | xargs -I % make -j%

make install

