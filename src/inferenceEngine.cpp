#include <omp.h>
#include "inferenceEngine.h"

EvalInferenceEngine::EvalInferenceEngine()
        : m_mxnetEngineType("MXNET_ENGINE_TYPE=NaiveEngine")
        , m_ompNumThreads("OMP_NUM_THREADS=1")
{
    cv::setNumThreads(0);
    putenv(m_ompNumThreads);
    putenv(m_mxnetEngineType);
    omp_set_num_threads(1);
}
