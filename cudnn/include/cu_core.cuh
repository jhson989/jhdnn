#pragma once

#include <iostream>
#include <vector>
#include <cudnn.h>


/***************************************************************
 * Debug code
 ***************************************************************/
#define cudnnErrChk(ans) { cudnnAssert((ans), __FILE__, __LINE__); }
#define cudaErrChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
void cudnnAssert(cudnnStatus_t code, const char *file, int line);
void cudaAssert(cudaError_t code, const char *file, int line);



/***************************************************************
 * Layer
 ***************************************************************/
class cuLayerFloat {

    private:

        cudnnTensorDescriptor_t desc_input;
        cudnnTensorDescriptor_t desc_output;
        size_t bytes_workspace;

        /** Host memory **/
        float* h_grad;
        float* h_input;
        float* h_output;

        /** Device memory **/
        float* d_workspace;
        float* d_grad;
        float* d_input;
        float* d_output;

    public:
        cuLayerFloat();
        ~cuLayerFloat();
        void forward(float* input);
        void backward(float* back_grad);
};