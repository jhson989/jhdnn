#include <jhdnn.cuh>



/***************************************************************************************
 * Forward implementation
 ****************************************************************************************/

template <typename T>
__global__ void __kernel_conv_forward_naive(
    T* input, T* filter, T* output,
    const int BATCH_NUM, 
    const int INPUT_C, const int INPUT_H,const int INPUT_W, 
    const int FILTER_H, const int FILTER_W, 
    const int PAD_H, const int PAD_W, 
    const int STRIDE_H, const int STRIDE_W, 
    const int OUTPUT_C, const int OUTPUT_H, const int OUTPUT_W
) {

    int batch = blockIdx.z * blockDim.z + threadIdx.z;
    int out_c = blockIdx.y * blockDim.y + threadIdx.y;
    int out_hw = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = out_hw/OUTPUT_W;
    int out_w = out_hw%OUTPUT_W;

    if (out_c<OUTPUT_C && out_hw<OUTPUT_H*OUTPUT_W) {

        T value = 0;
        int y = STRIDE_H*out_h-PAD_H;
        int x = STRIDE_W*out_w-PAD_W;
    
        for (int c=0; c<INPUT_C; c++) {
            for (int h=0;h<FILTER_H; h++) {
                for (int w=0;w<FILTER_W; w++) {
    
                    if ( (0<=(y+h)&&(y+h)<INPUT_H) && (0<=(x+w)&&(x+w)<INPUT_W)  ) {
                        value += filter[out_c*(INPUT_C*FILTER_H*FILTER_W) + c*(FILTER_H*FILTER_W) + h*(FILTER_W) + w] * input[batch*(INPUT_C*INPUT_H*INPUT_W) + c*(INPUT_H*INPUT_W) + (y+h)*(INPUT_W) + (x+w)];
                    }
    
                }
            }
        }
        
        output[batch*(OUTPUT_C*OUTPUT_H*OUTPUT_W) + out_c*(OUTPUT_H*OUTPUT_W) + out_h*(OUTPUT_W) + out_w] = value;
    }
}


/***************************************************************************************
 * Backward implementation
 ****************************************************************************************/

template <typename T>
__global__ void __kernel_conv_backward_naive(
    T* dy, T* filter, T* dx,
    const int BATCH_NUM, 
    const int INPUT_C, const int INPUT_H,const int INPUT_W, 
    const int FILTER_H, const int FILTER_W, 
    const int PAD_H, const int PAD_W, 
    const int STRIDE_H, const int STRIDE_W, 
    const int OUTPUT_C, const int OUTPUT_H, const int OUTPUT_W
) {

    int batch = blockIdx.z * blockDim.z + threadIdx.z;
    int in_c = blockIdx.y * blockDim.y + threadIdx.y;
    int in_hw = blockIdx.x * blockDim.x + threadIdx.x;
    int in_h = in_hw/INPUT_W;
    int in_w = in_hw%INPUT_W;

    if (in_c<INPUT_C && in_h<INPUT_H && in_w<INPUT_W) {

        T value = 0; 
        for (int c=0; c<OUTPUT_C; c++) {
            for (int h=-FILTER_H/2;h<=FILTER_H/2; h++) {
                for (int w=-FILTER_W/2;w<=FILTER_W/2; w++) {
                    //int y = ((in_h+PAD_H)%STRIDE_H != 0 ? (in_h+PAD_H)/STRIDE_H-1 : (in_h+PAD_H)/STRIDE_H) - h;
                    //int x = ((in_w+PAD_W)%STRIDE_W != 0 ? (in_w+PAD_W)/STRIDE_W-1 : (in_w+PAD_W)/STRIDE_W) - w;
                    //int y = (in_h+FILTER_H/2-PAD_H)/STRIDE_H - (h - FILTER_H/2);
                    //int x = (in_w+FILTER_W/2-PAD_W)/STRIDE_W - (w - FILTER_W/2);
                    //int y = (in_h-(h-FILTER_H/2)+PAD_H-FILTER_H/2)/STRIDE_H;
                    //int x = (in_w-(w-FILTER_W/2)+PAD_W-FILTER_W/2)/STRIDE_W;

                    //int y = (in_h +( (FILTER_H)/2 - PAD_H))/STRIDE_H - h;
                    //int x = (in_w +( (FILTER_W)/2 - PAD_W))/STRIDE_W - w;

                    int y = ((in_h - h) +FILTER_H/2 -PAD_H)/STRIDE_H;
                    int x = ((in_w - w) +FILTER_W/2 -PAD_W)/STRIDE_W;

                    
                    if (in_h == 1 && in_w ==0) 
                        printf("(%d, %d) (%d = %d - %d, %d = %d - %d)\n", y, x, y, (in_h +( (FILTER_H)/2 - PAD_H))/STRIDE_H, h, x, (in_w +( (FILTER_W)/2 - PAD_W))/STRIDE_W, w);
                    

                    if ( (0<=(y)&&(y)<OUTPUT_H) && (0<=(x)&&(x)<OUTPUT_W) ) {


                        value += (filter[c*(INPUT_C*FILTER_H*FILTER_W) + in_c*(FILTER_H*FILTER_W) + (h+FILTER_H/2)*(FILTER_W) + (w+FILTER_W/2)] * dy[batch*(OUTPUT_C*OUTPUT_H*OUTPUT_W) + c*(OUTPUT_H*OUTPUT_W) + y*(OUTPUT_W) + x]);

                        if (in_h == 1 && in_w == 0)
                            printf("out(%d %d) <- f(%d %d) %f %f\n", y, x, h, w, filter[c*(INPUT_C*FILTER_H*FILTER_W) + in_c*(FILTER_H*FILTER_W) + (h+FILTER_H/2)*(FILTER_W) + (w+FILTER_W/2)], value);

                    }

                }
            }
        }
        
        dx[batch*(INPUT_C*INPUT_H*INPUT_W) + in_c*(INPUT_H*INPUT_W) + in_h*(INPUT_W) + in_w] = value;
    }
}



/***************************************************************************************
 * Layer implementation
 ****************************************************************************************/

jhConvFloat::jhConvFloat(

    const int BATCH_NUM_, 
    const int INPUT_C_, const int INPUT_H_,const int INPUT_W_, 
    const int OUTPUT_C_,
    const int FILTER_H_, const int FILTER_W_, 
    const int PAD_H_, const int PAD_W_, 
    const int STRIDE_H_, const int STRIDE_W_,
    const int DILATION_H_, const int DILATION_W_

) : BATCH_NUM(BATCH_NUM_), INPUT_C(INPUT_C_), INPUT_H(INPUT_H_), INPUT_W(INPUT_W_), FILTER_H(FILTER_H_), FILTER_W(FILTER_W_), PAD_H(PAD_H_), PAD_W(PAD_W_), STRIDE_H(STRIDE_H_), STRIDE_W(STRIDE_W_), OUTPUT_C(OUTPUT_C_), DILATION_H(DILATION_H_), DILATION_W(DILATION_W_)
{
    OUTPUT_H=(INPUT_H-FILTER_H+2*PAD_H)/STRIDE_H + 1;
    OUTPUT_W=(INPUT_W-FILTER_W+2*PAD_W)/STRIDE_W + 1;
    printf("OUTPUT: [%d %d %d]\n", OUTPUT_C, OUTPUT_H, OUTPUT_W);
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

    int WARP_SIZE = 16;
    const dim3 dim_threads(WARP_SIZE, WARP_SIZE, 1);
    const dim3 dim_blocks((OUTPUT_H*OUTPUT_W+WARP_SIZE-1)/WARP_SIZE, (OUTPUT_C+WARP_SIZE-1)/WARP_SIZE, BATCH_NUM);
    cudaErrChk( cudaMemcpy(d_x, x, sizeof(float)*BATCH_NUM*INPUT_C*INPUT_H*INPUT_W, cudaMemcpyDeviceToDevice) );
    __kernel_conv_forward_naive<<<dim_blocks, dim_threads>>> (d_x, d_filter, d_y, BATCH_NUM, INPUT_C,INPUT_H,INPUT_W, FILTER_H,FILTER_W, PAD_H,PAD_W, STRIDE_H,STRIDE_W, OUTPUT_C,OUTPUT_H,OUTPUT_W);
    cudaErrChk( cudaDeviceSynchronize() );

}

void jhConvFloat::backward(float* dy) {

    int WARP_SIZE = 16;
    const dim3 dim_threads(WARP_SIZE, WARP_SIZE, 1);
    const dim3 dim_blocks((INPUT_H*INPUT_W+WARP_SIZE-1)/WARP_SIZE, (INPUT_C+WARP_SIZE-1)/WARP_SIZE, BATCH_NUM);
    cudaErrChk( cudaMemcpy(d_dy, dy, sizeof(float)*BATCH_NUM*OUTPUT_C*OUTPUT_H*OUTPUT_W, cudaMemcpyDeviceToDevice) );
    __kernel_conv_backward_naive<<<dim_blocks, dim_threads>>> (d_dy, d_filter, d_dx, BATCH_NUM, INPUT_C,INPUT_H,INPUT_W, FILTER_H,FILTER_W, PAD_H,PAD_W, STRIDE_H,STRIDE_W, OUTPUT_C,OUTPUT_H,OUTPUT_W);
    cudaErrChk( cudaDeviceSynchronize() );


}


void jhConvFloat::set_weights(float* filter_) {
    memcpy(h_filter, filter_, sizeof(float)*OUTPUT_C*INPUT_C*FILTER_H*FILTER_W);
    cudaErrChk( cudaMemcpy(d_filter, h_filter, sizeof(float)*OUTPUT_C*INPUT_C*FILTER_H*FILTER_W, cudaMemcpyHostToDevice) );
}
