#include <jhdnn.cuh>
#include <vector>
#include <algorithm>

int main(void) {

    
    /*******************************************************************************
     * Set input and filter
     ********************************************************************************/ 

    // Input configuration
    const int BATCH_NUM=3, INPUT_C=3, INPUT_H=127, INPUT_W=200;
    const int OUTPUT_C=7, FILTER_H=4, FILTER_W=4;
    const int PAD_H=2, PAD_W=1;
    const int STRIDE_H=5, STRIDE_W=4;
    int OUTPUT_H=(INPUT_H-FILTER_H+2*PAD_H)/STRIDE_H + 1;
    int OUTPUT_W=(INPUT_W-FILTER_W+2*PAD_W)/STRIDE_W + 1;

    // Input 
    float* d_input;
    std::vector<float> input(BATCH_NUM*INPUT_C*INPUT_H*INPUT_W);
    std::generate( input.begin(), input.end(), [](){ return ( (std::rand()%101-50)/10.0f); } );
    cudaErrChk( cudaMalloc(&d_input, sizeof(float)*BATCH_NUM*INPUT_C*INPUT_H*INPUT_W) );
    cudaErrChk( cudaMemcpy(d_input, input.data(), sizeof(float)*BATCH_NUM*INPUT_C*INPUT_H*INPUT_W, cudaMemcpyHostToDevice) );
    
    // Filter
    std::vector<float> filter(INPUT_C*OUTPUT_C*FILTER_H*FILTER_W);
    std::generate(filter.begin(), filter.end(), [](){return (std::rand()%101-50)/10.f;});



    /*******************************************************************************
     * Define cudnn and jhdnn convolution layer
     ********************************************************************************/ 
    cudnn_create();
    cuConvFloat conv_cu(
        BATCH_NUM, INPUT_C, INPUT_H, INPUT_W,
        OUTPUT_C, FILTER_H, FILTER_W,
        PAD_H, PAD_W,
        STRIDE_H, STRIDE_W
    );
    jhConvFloat conv_jh(
        BATCH_NUM, INPUT_C, INPUT_H, INPUT_W,
        OUTPUT_C, FILTER_H, FILTER_W,
        PAD_H, PAD_W,
        STRIDE_H, STRIDE_W
    );


    /*******************************************************************************
     * Convolution forward
     ********************************************************************************/ 
    conv_cu.set_weights(filter.data());
    conv_jh.set_weights(filter.data());

    conv_cu.forward(d_input);
    conv_jh.forward(d_input);


    /*******************************************************************************
     * Check forward results
     ********************************************************************************/ 

    std::vector<float> cu_y(BATCH_NUM*OUTPUT_C*OUTPUT_H*OUTPUT_W);
    cudaErrChk( cudaMemcpy(cu_y.data(), conv_cu.get_y(), sizeof(float)*BATCH_NUM*OUTPUT_C*OUTPUT_H*OUTPUT_W, cudaMemcpyDeviceToHost) );

    std::vector<float> jh_y(BATCH_NUM*OUTPUT_C*OUTPUT_H*OUTPUT_W);
    cudaErrChk( cudaMemcpy(jh_y.data(), conv_jh.get_y(), sizeof(float)*BATCH_NUM*OUTPUT_C*OUTPUT_H*OUTPUT_W, cudaMemcpyDeviceToHost) );
    cudaErrChk( cudaDeviceSynchronize() );  

    bool check = true;
    for (int i=0; i<cu_y.size(); i++) {
        if( cu_y[i] != jh_y[i] ) {
            check = false;
            printf("Forward : error!: cu[%d](%.3f) != jh[%d](%.3f)\n", i, cu_y[i], i, jh_y[i]);
            break;
        }
    }
    if (check) {
        printf("Forward : no error!\n");
    }
    


    /*******************************************************************************
     * Convolution backward
     ********************************************************************************/ 
    float* d_dy;
    std::vector<float> dy(BATCH_NUM*OUTPUT_C*OUTPUT_H*OUTPUT_W, 1);
    cudaErrChk( cudaMalloc(&d_dy, sizeof(float)*BATCH_NUM*OUTPUT_C*OUTPUT_H*OUTPUT_W) );
    cudaErrChk( cudaMemcpy(d_dy, dy.data(), sizeof(float)*BATCH_NUM*OUTPUT_C*OUTPUT_H*OUTPUT_W, cudaMemcpyHostToDevice) );
    cudaErrChk( cudaDeviceSynchronize() );

    conv_cu.backward(d_dy);
    conv_jh.backward(d_dy);

    /*******************************************************************************
     * Check backward dx results
     ********************************************************************************/ 
 
    std::vector<float> cu_dx(BATCH_NUM*INPUT_C*INPUT_H*INPUT_W);
    cudaErrChk( cudaMemcpy(cu_dx.data(), conv_cu.get_dx(), sizeof(float)*BATCH_NUM*INPUT_C*INPUT_H*INPUT_W, cudaMemcpyDeviceToHost) );

    std::vector<float> jh_dx(BATCH_NUM*INPUT_C*INPUT_H*INPUT_W);
    cudaErrChk( cudaMemcpy(jh_dx.data(), conv_jh.get_dx(), sizeof(float)*BATCH_NUM*INPUT_C*INPUT_H*INPUT_W, cudaMemcpyDeviceToHost) );
    cudaErrChk( cudaDeviceSynchronize() );  

    check = true;
    for (int i=0; i<cu_dx.size(); i++) {
        if( cu_dx[i] != jh_dx[i] ) {
            check = false;
            printf("Backward : error!: cu[%d](%.3f) != jh[%d](%.3f)\n", i, cu_dx[i], i, jh_dx[i]);
            break;
        }
    }
    if (check) {
        printf("Backward : no error!\n");
    }
     
 


    cudnn_destroy();
    return 0;       
}