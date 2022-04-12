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
        cudnnTensorDescriptor_t desc_x;
        cudnnTensorDescriptor_t desc_y;
        cudnnTensorDescriptor_t desc_dx;
        cudnnTensorDescriptor_t desc_dy;
        size_t bytes_workspace_forward;
        size_t bytes_workspace_backward_data;

        /** Host memory **/
        float* h_dx;
        float* h_dy;
        float* h_x;
        float* h_y;

        /** Device memory **/
        float* d_workspace_forward;
        float* d_workspace_backward_data;
        float* d_dx;
        float* d_dy;
        float* d_x;
        float* d_y;

    public:
        cuLayerFloat(){};
        ~cuLayerFloat(){};
        virtual void forward(float* x){};
        virtual void backward(float* dy){};
        virtual float* get_y() {return d_y;};
};