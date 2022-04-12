#include <jhdnn.cuh>
#include <vector>
#include <algorithm>

int main(void) {


    cudnn_create();
    cuConvFloat conv_cu(
        3, 3, 128, 128,
        3, 128, 128,
        3, 3
    );
    jhConvFloat conv_jh(
        3, 3, 128, 128,
        3, 128, 128,
        3, 3
    );

    // filter
    float filter[] = {
        -1, -1, -1, -1, 9, -1, -1, -1, -1,
        1, 0, 0, 0, 0, 0, 0, 4, 0,
        1, 0, 0, 0, 0, 0, 0, 4, 0,
        1, 0, 0, 0, 0, 0, 0, 4, 0,
        -1, -1, -1, -1, 9, -1, -1, -1, -1,
        1, 0, 0, 0, 0, 0, 0, 4, 0,
        1, 0, 0, 0, 0, 0, 0, 4, 0,
        1, 0, 0, 0, 0, 0, 0, 4, 0,
        -1, -1, -1, -1, 9, -1, -1, -1, -1
    };



    /*******************************************************************************
     * Forward
     ********************************************************************************/ 

    // input
    float* d_input;
    std::vector<float> input(3*3*128*128, 0);
    std::generate( input.begin(), input.end(), [](){ return ( (std::rand()%101-50)/10.0f); } );
    cudaErrChk( cudaMalloc(&d_input, sizeof(float)*3*3*128*128) );
    cudaErrChk( cudaMemcpy(d_input, input.data(), sizeof(float)*3*3*128*128, cudaMemcpyHostToDevice) );
    
    conv_cu.set_weights(filter);
    conv_cu.forward(d_input);
    conv_jh.set_weights(filter);
    conv_jh.forward(d_input);

    /*******************************************************************************
     * Check result
     ********************************************************************************/ 
    float* cu_d_result = conv_cu.get_y();
    float* jh_d_result = conv_jh.get_y();

    std::vector<float> cu_y(3*3*128*128);
    cudaErrChk( cudaMemcpy(cu_y.data(), cu_d_result, sizeof(float)*3*3*128*128, cudaMemcpyDeviceToHost) );

    std::vector<float> jh_y(3*3*128*128);
    cudaErrChk( cudaMemcpy(jh_y.data(), jh_d_result, sizeof(float)*3*3*128*128, cudaMemcpyDeviceToHost) );
    cudaErrChk( cudaDeviceSynchronize() );  

    bool check = true;
    for (int i=0; i<cu_y.size(); i++) {
        if( cu_y[i] != jh_y[i] ) {
            check = false;
            printf("error!: cu[%d](%.3f) != jh[%d](%.3f)\n", i, cu_y[i], i, jh_y[i]);
            break;
        }
    }
    if (check) {
        printf("No error!\n");
    }
    


    /*******************************************************************************
     * Backward
     ********************************************************************************/ 

    float* d_dy;
    std::vector<float> dy(3*3*128*128, 1);
    cudaErrChk( cudaMalloc(&d_dy, sizeof(float)*3*3*128*128) );
    cudaErrChk( cudaMemcpy(d_dy, dy.data(), sizeof(float)*3*3*128*128, cudaMemcpyHostToDevice) );
    cudaErrChk( cudaDeviceSynchronize() );


    conv_cu.backward(d_dy);
    conv_jh.backward(d_dy);





    cudnn_destroy();
    return 0;       
}