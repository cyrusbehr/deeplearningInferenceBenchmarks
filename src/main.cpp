#include <iostream>
#include "inferenceManager.h"
#include <stdlib.h>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " /path/to/model/directory/ num_threads (optional)" << std::endl;
        return -1;
    }

    if (argc == 3) {
        std::string name = "OMP_NUM_THREADS";
        std::string value = argv[2];
        setenv(name.c_str(), value.c_str(), 1);
    }

    InferenceManager inferenceManager(argv[1]);
    inferenceManager.runBenchmark(1);

    return 0;
}
