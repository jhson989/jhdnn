#pragma once

#include <jh_core.cuh>

class jhConvFloat : public jhLayerFloat {

    private:

        /** Data **/
        cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
        const int BATCH_NUM;
        const int INPUT_C; const int INPUT_H; const int INPUT_W;
        const int OUTPUT_C; const int OUTPUT_H; const int OUTPUT_W;
        const int FILTER_H; const int FILTER_W;
        const int PAD_H; const int PAD_W;
        const int STRIDE_H; const int STRIDE_W; 
        const int DILATION_H; const int DILATION_W;

        /** Host memory **/
        float* h_filter;

        /** Device memory **/
        float* d_filter;
        float* d_dw;
    

    public:
        jhConvFloat(
            const int BATCH_NUM, 
            const int INPUT_C, const int INPUT_H,const int INPUT_W, 
            const int OUTPUT_C, const int OUTPUT_H, const int OUTPUT_W,
            const int FILTER_H, const int FILTER_W, 
            const int PAD_H=1, const int PAD_W=1, 
            const int STRIDE_H=1, const int STRIDE_W=1, 
            const int DILATION_H=1, const int DILATION_W=1
        );

        ~jhConvFloat();

        virtual void forward(float* x) override;
        virtual void backward(float* dy) override;
        void set_weights(float* filter_);


};