#pragma once
#include <cudnn.h>
#include <cu_core.cuh>

class cuLinearFloat : public cuLayerFloat {

    private:

        const int BATCH_NUM;
        const int IN_FEATURES;
        const int OUT_FEATURES;

        cudnnBackendDescriptor_t desc_matmul;
        cudnnBackendDescriptor_t desc_A;
        cudnnBackendDescriptor_t desc_B;
        cudnnBackendDescriptor_t desc_C;

    public:
        cuLinearFloat(const int BATCH_NUM, const int IN_FEATURES, const int OUT_FEATURES);
        ~cuLinearFloat();
};