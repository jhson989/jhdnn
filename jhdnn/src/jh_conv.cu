#include <jhdnn.cuh>


jhConvFloat::jhConvFloat(

    const int BATCH_NUM_, 
    const int INPUT_C_, const int INPUT_H_,const int INPUT_W_, 
    const int OUTPUT_C_, const int OUTPUT_H_, const int OUTPUT_W_,
    const int FILTER_H_, const int FILTER_W_, 
    const int PAD_H_, const int PAD_W_, 
    const int STRIDE_H_, const int STRIDE_W_,
    const int DILATION_H_, const int DILATION_W_

) : BATCH_NUM(BATCH_NUM_), INPUT_C(INPUT_C_), INPUT_H(INPUT_H_), INPUT_W(INPUT_W_), FILTER_H(FILTER_H_), FILTER_W(FILTER_W_), PAD_H(PAD_H_), PAD_W(PAD_W_), STRIDE_H(STRIDE_H_), STRIDE_W(STRIDE_W_), OUTPUT_C(OUTPUT_C_), OUTPUT_H(OUTPUT_H_), OUTPUT_W(OUTPUT_W_), DILATION_H(DILATION_H_), DILATION_W(DILATION_W_)
{

    /******************************************************************
     * 1. Allocate device memory
     *******************************************************************/

    h_filter = (float*) malloc(sizeof(float)*OUTPUT_C*INPUT_C*FILTER_H*FILTER_W);
    cudaErrChk( cudaMalloc(&d_x, sizeof(float)*BATCH_NUM*INPUT_C*INPUT_H*INPUT_W) );
    cudaErrChk( cudaMalloc(&d_y, sizeof(float)*BATCH_NUM*OUTPUT_C*OUTPUT_H*OUTPUT_W) );
    cudaErrChk( cudaMalloc(&d_dx, sizeof(float)*BATCH_NUM*INPUT_C*INPUT_H*INPUT_W) );
    cudaErrChk( cudaMalloc(&d_dy, sizeof(float)*BATCH_NUM*OUTPUT_C*OUTPUT_H*OUTPUT_W) );
    cudaErrChk( cudaMalloc(&d_filter, sizeof(float)*OUTPUT_C*INPUT_C*FILTER_H*FILTER_W) );
    cudaErrChk( cudaMalloc(&d_dw, sizeof(float)*OUTPUT_C*INPUT_C*FILTER_H*FILTER_W) );

    /******************************************************************
     * 2. Initialize filter
     *******************************************************************/
    std::generate(h_filter, h_filter+OUTPUT_C*INPUT_C*FILTER_H*FILTER_W, [](){return (std::rand()%101-50)/10;});
    cudaErrChk( cudaMemcpy(d_filter, h_filter, sizeof(float)*OUTPUT_C*INPUT_C*FILTER_H*FILTER_W, cudaMemcpyHostToDevice) );

}


jhConvFloat::~jhConvFloat() {
    free (h_filter);
    cudaErrChk( cudaFree(d_x) );
    cudaErrChk( cudaFree(d_y) );
    cudaErrChk( cudaFree(d_dx) );
    cudaErrChk( cudaFree(d_dy) );
    cudaErrChk( cudaFree(d_filter) );

}


void jhConvFloat::forward(float* x) {

}

void jhConvFloat::backward(float* dy) {
}


void jhConvFloat::set_weights(float* filter_) {
    memcpy(h_filter, filter_, sizeof(float)*OUTPUT_C*INPUT_C*FILTER_H*FILTER_W);
    cudaErrChk( cudaMemcpy(d_filter, h_filter, sizeof(float)*OUTPUT_C*INPUT_C*FILTER_H*FILTER_W, cudaMemcpyHostToDevice) );
}