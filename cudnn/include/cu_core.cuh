#pragma once

#include <cstring>
#include <algorithm>
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
void cudnn_create();
void cudnn_destroy();


/***************************************************************
 * Layer
 ***************************************************************/
class cuLayerFloat {

    protected:
        cudnnTensorDescriptor_t desc_input;
        cudnnTensorDescriptor_t desc_output;
        cudnnTensorDescriptor_t desc_dx;
        cudnnTensorDescriptor_t desc_dy;
        size_t bytes_workspace_forward;
        size_t bytes_workspace_backward_data;

        /** Host memory **/
        float* h_dx;
        float* h_dy;
        float* h_input;
        float* h_output;

        /** Device memory **/
        float* d_workspace_forward;
        float* d_workspace_backward_data;
        float* d_dx;
        float* d_dy;
        float* d_input;
        float* d_output;

    public:
        cuLayerFloat(){};
        ~cuLayerFloat(){};
        virtual void forward(float* input){};
        virtual void backward(float* dy){};
};