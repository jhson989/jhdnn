#include <jhdnn.cuh>
#include <vector>
#include <algorithm>

int main(void) {


    // input
    float* d_input;
    std::vector<float> input(3*1*128*128, 0);
    std::generate( input.begin(), input.end(), [](){ return ((std::rand()%101-50)%10); } );
    cudaErrChk( cudaMalloc(&d_input, sizeof(float)*3*1*128*128) );
    cudaErrChk( cudaMemcpy(d_input, input.data(), sizeof(float)*3*1*128*128, cudaMemcpyHostToDevice) );
    
    cudnn_create();
    cuConvFloat conv(
        3, 3, 128, 128,
        3, 128, 128,
        3, 3
    );

    // filter
    float filter[] = {
        -1, -1, -1, -1, 9, -1, -1, -1, -1,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        -1, -1, -1, -1, 9, -1, -1, -1, -1,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        -1, -1, -1, -1, 9, -1, -1, -1, -1
    };
    conv.set_weights(filter);

    float* d_dy;
    std::vector<float> dy(3*1*128*128, 1);
    cudaErrChk( cudaMalloc(&d_dy, sizeof(float)*3*1*128*128) );
    cudaErrChk( cudaMemcpy(d_dy, dy.data(), sizeof(float)*3*1*128*128, cudaMemcpyHostToDevice) );
    cudaErrChk( cudaDeviceSynchronize() );

    conv.forward(d_input);
    conv.backward(d_dy);

    cudnn_destroy();
    return 0;       
}