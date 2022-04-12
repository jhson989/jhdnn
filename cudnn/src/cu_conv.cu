#include <jhdnn.cuh>

extern cudnnHandle_t* cudnn;


cuConvFloat::cuConvFloat(

    const int BATCH_NUM_, 
    const int INPUT_C_, const int INPUT_H_, const int INPUT_W_, 
    const int OUTPUT_C_, const int FILTER_H_, const int FILTER_W_, 
    const int PAD_H_, const int PAD_W_, 
    const int STRIDE_H_, const int STRIDE_W_,
    const int DILATION_H_, const int DILATION_W_
    
) : BATCH_NUM(BATCH_NUM_), INPUT_C(INPUT_C_), INPUT_H(INPUT_H_), INPUT_W(INPUT_W_), OUTPUT_C(OUTPUT_C_), FILTER_H(FILTER_H_), FILTER_W(FILTER_W_), PAD_H(PAD_H_), PAD_W(PAD_W_), STRIDE_H(STRIDE_H_), STRIDE_W(STRIDE_W_), DILATION_H(DILATION_H_), DILATION_W(DILATION_W_)
{
    OUTPUT_H=(INPUT_H-FILTER_H+2*PAD_H)/STRIDE_H + 1;
    OUTPUT_W=(INPUT_W-FILTER_W+2*PAD_W)/STRIDE_W + 1;
    /******************************************************************
     * 2. Describe Conv2D operands
     *    - Input tensor : size, layout
     *    - Output tensor : size, layout
     *    - dx tensor : size, layout
     *    - dy tensor : size, layout
     *******************************************************************/
    cudnnErrChk( cudnnCreateTensorDescriptor(&desc_x) );
    cudnnErrChk( cudnnSetTensor4dDescriptor(
        desc_x,
        /*LAYOUT*/CUDNN_TENSOR_NCHW, /*DATATYPE*/data_type, /*N*/BATCH_NUM, /*C*/ INPUT_C, /*H*/INPUT_H, /*W*/INPUT_W
    ) );

    cudnnErrChk( cudnnCreateTensorDescriptor(&desc_y) );
    cudnnErrChk( cudnnSetTensor4dDescriptor(
        desc_y,
        /*LAYOUT*/CUDNN_TENSOR_NCHW, /*DATATYPE*/data_type, /*N*/BATCH_NUM, /*C*/ OUTPUT_C, /*H*/OUTPUT_H, /*W*/OUTPUT_W
    ) );

    cudnnErrChk( cudnnCreateTensorDescriptor(&desc_dx) );
    cudnnErrChk( cudnnSetTensor4dDescriptor(
        desc_dx,
        /*LAYOUT*/CUDNN_TENSOR_NCHW, /*DATATYPE*/data_type, /*N*/BATCH_NUM, /*C*/ INPUT_C, /*H*/INPUT_H, /*W*/INPUT_W
    ) );

    cudnnErrChk( cudnnCreateTensorDescriptor(&desc_dy) );
    cudnnErrChk( cudnnSetTensor4dDescriptor(
        desc_dy,
        /*LAYOUT*/CUDNN_TENSOR_NCHW, /*DATATYPE*/data_type, /*N*/BATCH_NUM, /*C*/ OUTPUT_C, /*H*/OUTPUT_H, /*W*/OUTPUT_W
    ) );


    /******************************************************************
     * 3. Describe Conv2D kernel
     *    - Filter : layout, size
     *    - Conv2D layer : pad, stride, dilation, etc.
     *    - Conv2d forward algorithm
     *    - Conv2d backward algorithm
     *******************************************************************/
    // Filter (weights)
    cudnnErrChk( cudnnCreateFilterDescriptor(&desc_filter) );
    cudnnErrChk( cudnnSetFilter4dDescriptor(
        desc_filter,
        /*DATATYPE*/data_type, /*LAYOUT*/CUDNN_TENSOR_NCHW, /*OUT_CH*/OUTPUT_C, /*IN_CH*/ INPUT_C, /*KERNEL_H*/FILTER_H, /*KERNEL_W*/FILTER_W
    ) );

    // Filter (weights) backward
    cudnnErrChk( cudnnCreateFilterDescriptor(&desc_dw) );
    cudnnErrChk( cudnnSetFilter4dDescriptor(
        desc_dw,
        /*DATATYPE*/data_type, /*LAYOUT*/CUDNN_TENSOR_NCHW, /*OUT_CH*/OUTPUT_C, /*IN_CH*/ INPUT_C, /*KERNEL_H*/FILTER_H, /*KERNEL_W*/FILTER_W
    ) );

    // Layer 
    cudnnErrChk( cudnnCreateConvolutionDescriptor(&desc_conv2d) );
    cudnnErrChk( cudnnSetConvolution2dDescriptor(
        desc_conv2d,
        /*PAD_H*/PAD_H, /*PAD_W*/PAD_W, /*STRIDE_VERTICAL*/STRIDE_H, /*STRIDE_HORIZONTAL*/STRIDE_W, /*DILATION_H*/DILATION_H, /*DILATION_W*/DILATION_W, /*MODE*/CUDNN_CROSS_CORRELATION, /*DATATYPE*/data_type
    ) );

    // Forward algorithm
    cudnnErrChk( cudnnFindConvolutionForwardAlgorithm(
        *cudnn, desc_x, desc_filter, desc_conv2d, desc_y, 1, &num_conv2d_algo_forward, &perf_conv2d_algo_forward
    ) );

    // Backward algorithm
    cudnnErrChk( cudnnFindConvolutionBackwardDataAlgorithm(
        *cudnn, desc_filter, desc_dy, desc_conv2d, desc_dx, 1, &num_conv2d_algo_backward_data, &perf_conv2d_algo_backward_data
    ) );
    cudnnErrChk( cudnnFindConvolutionBackwardFilterAlgorithm(
        *cudnn, desc_x, desc_dy, desc_conv2d, desc_dw, 1, &num_conv2d_algo_backward_filter, &perf_conv2d_algo_backward_filter
    ) );

    /******************************************************************
     * 4. Calculate work-space size for forward and backward
     *******************************************************************/
    cudnnErrChk( cudnnGetConvolutionForwardWorkspaceSize(*cudnn, desc_x, desc_filter, desc_conv2d, desc_y, perf_conv2d_algo_forward.algo, &bytes_workspace_forward) );
    cudnnErrChk( cudnnGetConvolutionBackwardDataWorkspaceSize(*cudnn, desc_filter, desc_dy, desc_conv2d, desc_dx, perf_conv2d_algo_backward_data.algo, &bytes_workspace_backward_data) );
    cudnnErrChk( cudnnGetConvolutionBackwardFilterWorkspaceSize(*cudnn, desc_x, desc_dy, desc_conv2d, desc_dw, perf_conv2d_algo_backward_filter.algo, &bytes_workspace_backward_filter) );
    
    /******************************************************************
     * 5. Allocate memory
     *    - work-space
     *    - HOST : x, y, dx, dy, kernel -> Not necessary
     *    - GPU : x, y, dx, dy, kernel
     *******************************************************************/
    cudaErrChk (cudaMalloc (&d_workspace_forward, bytes_workspace_forward));
    cudaErrChk (cudaMalloc (&d_workspace_backward_data, bytes_workspace_backward_data));
    cudaErrChk (cudaMalloc (&d_workspace_backward_filter, bytes_workspace_backward_filter));
 
    h_filter = (float*) malloc(sizeof(float)*OUTPUT_C*INPUT_C*FILTER_H*FILTER_W);
    cudaErrChk( cudaMalloc(&d_x, sizeof(float)*BATCH_NUM*INPUT_C*INPUT_H*INPUT_W) );
    cudaErrChk( cudaMalloc(&d_y, sizeof(float)*BATCH_NUM*OUTPUT_C*OUTPUT_H*OUTPUT_W) );
    cudaErrChk( cudaMalloc(&d_dx, sizeof(float)*BATCH_NUM*INPUT_C*INPUT_H*INPUT_W) );
    cudaErrChk( cudaMalloc(&d_dy, sizeof(float)*BATCH_NUM*OUTPUT_C*OUTPUT_H*OUTPUT_W) );
    cudaErrChk( cudaMalloc(&d_filter, sizeof(float)*OUTPUT_C*INPUT_C*FILTER_H*FILTER_W) );
    cudaErrChk( cudaMalloc(&d_dw, sizeof(float)*OUTPUT_C*INPUT_C*FILTER_H*FILTER_W) );
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
    free (h_filter);

    cudaErrChk( cudaFree(d_workspace_forward) );
    cudaErrChk( cudaFree(d_workspace_backward_data) );
    cudaErrChk( cudaFree(d_workspace_backward_filter) );
    
    cudaErrChk( cudaFree(d_x) );
    cudaErrChk( cudaFree(d_y) );
    cudaErrChk( cudaFree(d_dx) );
    cudaErrChk( cudaFree(d_dy) );
    cudaErrChk( cudaFree(d_filter) );

    cudnnErrChk( cudnnDestroyTensorDescriptor(desc_x) );
    cudnnErrChk( cudnnDestroyTensorDescriptor(desc_y) );
    cudnnErrChk( cudnnDestroyTensorDescriptor(desc_dx) );
    cudnnErrChk( cudnnDestroyTensorDescriptor(desc_dy) );

    cudnnErrChk( cudnnDestroyFilterDescriptor(desc_filter) );
    cudnnErrChk( cudnnDestroyFilterDescriptor(desc_dw) );

}


void cuConvFloat::forward(float* x) {

    /******************************************************************
     * 6. Launch forward kernel
     *******************************************************************/
    const float alpha=1, beta=0;
    
    cudaErrChk( cudaMemcpy(d_x, x, sizeof(float)*BATCH_NUM*INPUT_C*INPUT_H*INPUT_W, cudaMemcpyDeviceToDevice) );

    cudnnErrChk( cudnnConvolutionForward(*cudnn
                                        , /*ALPHA*/&alpha
                                        , /*INPUT*/desc_x, d_x
                                        , /*KERNEL*/desc_filter, d_filter
                                        , /*LAYER*/desc_conv2d, perf_conv2d_algo_forward.algo, d_workspace_forward, bytes_workspace_forward
                                        , /*BETA*/&beta
                                        , /*OUTPUT*/desc_y, d_y
                                    ) );
    cudaErrChk( cudaDeviceSynchronize() );

}

void cuConvFloat::backward(float* dy) {

    /******************************************************************
     * 6. Launch backward kernel
     *******************************************************************/
    
    const float alpha=1, beta=0;
    cudnnErrChk( cudnnConvolutionBackwardData(*cudnn
                                        , /*ALPHA*/&alpha
                                        , /*KERNEL*/desc_filter, d_filter
                                        , /*dy*/desc_dy, d_dy
                                        , /*LAYER*/desc_conv2d, perf_conv2d_algo_backward_data.algo, d_workspace_backward_data, bytes_workspace_backward_data
                                        , /*BETA*/&beta
                                        , /*dx*/desc_dx, d_dx
                                    ) );
    
    cudnnErrChk( cudnnConvolutionBackwardFilter(*cudnn
                                        , /*ALPHA*/&alpha
                                        , /*x*/desc_x, d_x
                                        , /*dy*/desc_dy, d_dy
                                        , /*LAYER*/desc_conv2d, perf_conv2d_algo_backward_filter.algo, d_workspace_backward_filter, bytes_workspace_backward_filter
                                        , /*BETA*/&beta
                                        , /*dw*/desc_dw, d_dw
                                    ) );
    
    cudaErrChk( cudaDeviceSynchronize() );
}

void cuConvFloat::set_weights(float* filter_) {
    memcpy(h_filter, filter_, sizeof(float)*OUTPUT_C*INPUT_C*FILTER_H*FILTER_W);
    cudaErrChk( cudaMemcpy(d_filter, h_filter, sizeof(float)*OUTPUT_C*INPUT_C*FILTER_H*FILTER_W, cudaMemcpyHostToDevice) );
}
