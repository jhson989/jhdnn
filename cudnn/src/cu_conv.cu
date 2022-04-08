#include <jhdnn.cuh>

extern cudnnHandle_t* cudnn;


cuConvFloat::cuConvFloat(

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
     * 1. Define cudnn Handler
     *******************************************************************/
    if (cudnn == nullptr) {
        cudnn = new cudnnHandle_t;
        cudnnErrChk( cudnnCreate(cudnn) );
    }


    /******************************************************************
     * 2. Describe Conv2D operands
     *    - Input tensor : size, layout
     *    - Output tensor : size, layout
     *******************************************************************/
    cudnnErrChk( cudnnCreateTensorDescriptor(&desc_input) );
    cudnnErrChk( cudnnSetTensor4dDescriptor(
        desc_input,
        /*LAYOUT*/CUDNN_TENSOR_NCHW, /*DATATYPE*/data_type, /*N*/BATCH_NUM, /*C*/ INPUT_C, /*H*/INPUT_H, /*W*/INPUT_W
    ) );

    cudnnErrChk( cudnnCreateTensorDescriptor(&desc_output) );
    cudnnErrChk( cudnnSetTensor4dDescriptor(
        desc_output,
        /*LAYOUT*/CUDNN_TENSOR_NCHW, /*DATATYPE*/data_type, /*N*/BATCH_NUM, /*C*/ OUTPUT_C, /*H*/OUTPUT_H, /*W*/OUTPUT_W
    ) );




    /******************************************************************
     * 3. Describe Conv2D kernel
     *    - Filter : layout, size
     *    - Conv2D layer : pad, stride, dilation, etc.
     *    - Conv2d algorithm
     *******************************************************************/
    // Filter (weights)
    cudnnErrChk( cudnnCreateFilterDescriptor(&desc_filter) );
    cudnnErrChk( cudnnSetFilter4dDescriptor(
        desc_filter,
        /*DATATYPE*/data_type, /*LAYOUT*/CUDNN_TENSOR_NCHW, /*OUT_CH*/OUTPUT_C, /*IN_CH*/ INPUT_C, /*KERNEL_H*/FILTER_H, /*KERNEL_W*/FILTER_W
    ) );

    // Layer 
    cudnnErrChk( cudnnCreateConvolutionDescriptor(&desc_conv2d) );
    cudnnErrChk( cudnnSetConvolution2dDescriptor(
        desc_conv2d,
        /*PAD_H*/PAD_H, /*PAD_W*/PAD_W, /*STRIDE_VERTICAL*/STRIDE_H, /*STRIDE_HORIZONTAL*/STRIDE_W, /*DILATION_H*/DILATION_H, /*DILATION_W*/DILATION_W, /*MODE*/CUDNN_CROSS_CORRELATION, /*DATATYPE*/data_type
    ) );

    // Algorithm
    cudnnErrChk( cudnnFindConvolutionForwardAlgorithm(
        *cudnn, desc_input, desc_filter, desc_conv2d, desc_output, 1, &num_conv2d_algo, &perf_conv2d_algo
    ) );


    /******************************************************************
     * 4. Calculate work-space size
     *******************************************************************/
    cudnnErrChk( cudnnGetConvolutionForwardWorkspaceSize(*cudnn, desc_input, desc_filter, desc_conv2d, desc_output, perf_conv2d_algo.algo, &bytes_workspace) );
  

    /******************************************************************
     * 5. Allocate memory
     *    - work-space
     *    - HOST : input, output, kernel
     *    - GPU : input, output, kernel
     *******************************************************************/
    cudaErrChk (cudaMalloc (&d_workspace, bytes_workspace));
 
    h_input = (float*) malloc(sizeof(float)*BATCH_NUM*INPUT_C*INPUT_H*INPUT_W);
    h_output = (float*) malloc(sizeof(float)*BATCH_NUM*OUTPUT_C*OUTPUT_H*OUTPUT_W);
    h_filter = (float*) malloc(sizeof(float)*OUTPUT_C*INPUT_C*FILTER_H*FILTER_W);
    cudaErrChk( cudaMalloc(&d_input, sizeof(float)*BATCH_NUM*INPUT_C*INPUT_H*INPUT_W) );
    cudaErrChk( cudaMalloc(&d_output, sizeof(float)*BATCH_NUM*OUTPUT_C*OUTPUT_H*OUTPUT_W) );
    cudaErrChk( cudaMalloc(&d_filter, sizeof(float)*OUTPUT_C*INPUT_C*FILTER_H*FILTER_W) );

    /******************************************************************
     * 6. Initialize filter
     *******************************************************************/
    std::generate(h_filter, h_filter+OUTPUT_C*INPUT_C*FILTER_H*FILTER_W, [](){return (std::rand()%101-50)/10;});
    cudaErrChk( cudaMemcpy(d_filter, h_filter, sizeof(float)*OUTPUT_C*INPUT_C*FILTER_H*FILTER_W, cudaMemcpyHostToDevice) );
}




cuConvFloat::~cuConvFloat() {

    /******************************************************************
     * Finallize
     *******************************************************************/
     free (h_input);
     free (h_output);
     free (h_filter);
 
     cudaErrChk( cudaFree(d_workspace) );
     cudaErrChk( cudaFree(d_input) );
     cudaErrChk( cudaFree(d_output) );
     cudaErrChk( cudaFree(d_filter) );
 
     cudnnErrChk( cudnnDestroyTensorDescriptor(desc_input) );
     cudnnErrChk( cudnnDestroyTensorDescriptor(desc_output) );
     cudnnErrChk( cudnnDestroyFilterDescriptor(desc_filter) );

}


void cuConvFloat::forward(float* input) {

    /******************************************************************
     * 6. Launch kernel
     *******************************************************************/
    const float alpha=1, beta=0;
    cudnnErrChk( cudnnConvolutionForward(*cudnn
                                        , /*ALPHA*/&alpha
                                        , /*INPUT*/desc_input, d_input
                                        , /*KERNEL*/desc_filter, d_filter
                                        , /*LAYER*/desc_conv2d, perf_conv2d_algo.algo, d_workspace, bytes_workspace
                                        , /*BETA*/&beta
                                        , /*OUTPUT*/desc_output, d_output
                                    ) );
 


    /******************************************************************
     * 7. Get result
     *******************************************************************/
    cudaErrChk( cudaMemcpy(h_output, d_output, sizeof(float)*BATCH_NUM*OUTPUT_C*OUTPUT_H*OUTPUT_W, cudaMemcpyDeviceToHost) );

}

void cuConvFloat::backward(float* back_grad) {

}

void cuConvFloat::set_weight(float* filter_) {
    memcpy(h_filter, filter_, sizeof(float)*OUTPUT_C*INPUT_C*FILTER_H*FILTER_W);
    cudaErrChk( cudaMemcpy(d_filter, h_filter, sizeof(float)*OUTPUT_C*INPUT_C*FILTER_H*FILTER_W, cudaMemcpyHostToDevice) );
}