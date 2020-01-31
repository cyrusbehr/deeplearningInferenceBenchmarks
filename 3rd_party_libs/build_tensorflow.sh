test -e tensorflow_cc || git clone https://github.com/FloopCZ/tensorflow_cc.git
export CC_OPT_FLAGS="-march=native"
cd tensorflow_cc
cd tensorflow_cc
mkdir build && cd build
cmake -DTENSORFLOW_STATIC=OFF -DTENSORFLOW_SHARED=ON ..
make && sudo make install