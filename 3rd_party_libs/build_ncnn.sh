# download ncnn

test -e ncnn || git clone git@github.com:Tencent/ncnn.git

cd ncnn
git reset --hard 0e83b0f
#########

# build for broadwell architecture in single thread mode
#########
mkdir build_broadwell
cd build_broadwell

	cmake -D NCNN_BUILD_TOOLS=OFF -D NCNN_VULKAN=OFF -D CMAKE_BUILD_TYPE=Release -D NCNN_OPENMP=OFF -D NCNN_AVX2=ON -D CMAKE_CXX_FLAGS="-march=broadwell" ..

if [ "$(uname)" == "Darwin" ]; then
    sysctl -n hw.physicalcpu | xargs -I % make -j%
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    nproc | xargs -I % make -j%
fi

make install
