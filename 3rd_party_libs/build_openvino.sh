test -e openvino || git clone https://github.com/openvinotoolkit/openvino
cd openvino
git checkout tags/2021.2
git reset HEAD --hard
git submodule update --init --recursive

chmod +x install_build_dependencies.sh
./install_build_dependencies.sh

test -e build || mkdir build
cd build

cmake -DCMAKE_BUILD_TYPE=Release ..
make --jobs=$(nproc --all)
