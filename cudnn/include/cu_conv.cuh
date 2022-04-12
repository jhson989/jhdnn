#pragma once
#include <cudnn.h>
#include <cu_core.cuh>

class cuConvFloat : public cuLayerFloat {

    private:

        /** Data **/
        cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
        const int BATCH_NUM;
        const int INPUT_C; const int INPUT_H; const int INPUT_W;
        const int FILTER_H; const int FILTER_W;
        const int PAD_H; const int PAD_W;
        const int STRIDE_H; const int STRIDE_W; 
        const int DILATION_H; const int DILATION_W;
        const int OUTPUT_C; int OUTPUT_H; int OUTPUT_W;

        /** Convolution forward **/
        cudnnFilterDescriptor_t desc_filter;
        cudnnFilterDescriptor_t desc_dw;
        cudnnConvolutionDescriptor_t desc_conv2d;
        int num_conv2d_algo_forward;
        cudnnConvolutionFwdAlgoPerf_t perf_conv2d_algo_forward;

        /** Convolution backward **/
        int num_conv2d_algo_backward_data;
        cudnnConvolutionBwdDataAlgoPerf_t perf_conv2d_algo_backward_data;
        int num_conv2d_algo_backward_filter;
        cudnnConvolutionBwdFilterAlgoPerf_t perf_conv2d_algo_backward_filter;

        /** Host memory **/
        float* h_filter;

        /** Device memory **/
        float* d_filter;
        float* d_dw;
        size_t bytes_workspace_backward_filter;
        float* d_workspace_backward_filter;

    public:
        cuConvFloat(
            const int BATCH_NUM, 
            const int INPUT_C, const int INPUT_H,const int INPUT_W, 
            const int OUTPUT_C, const int FILTER_H, const int FILTER_W, 
            const int PAD_H=1, const int PAD_W=1, 
            const int STRIDE_H=1, const int STRIDE_W=1, 
            const int DILATION_H=1, const int DILATION_W=1
            
        );

        ~cuConvFloat();

        virtual void forward(float* x) override;
        virtual void backward(float* dy) override;
        void set_weights(float* filter_);
        float* get_dw() {return d_dw;};


};
