# Download ncnn
test -e ncnn || git clone --recursive https://github.com/Tencent/ncnn.git

cd ncnn
git reset --hard 5e4ea0b # Jan 24 release

mkdir build_amd64
cd build_amd64



#nproc | xargs -I % make -j%
#
#make install
#
