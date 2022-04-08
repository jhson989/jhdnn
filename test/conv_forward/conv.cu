

#include <jhdnn.cuh>
#include <vector>
#include <algorithm>

int main(void) {

    cuConvFloat conv(
        128, 3, 128, 128,
        3, 128, 128,
        3, 3
    );

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

    float* d_input;
    std::vector<float> input(128*1*128*128, 0);
    std::generate( input.begin(), input.end(), [](){ return ((std::rand()%101-50)%10); } );
    cudaErrChk( cudaMalloc(&d_input, sizeof(float)*128*1*128*128) );
    cudaErrChk( cudaMemcpy(d_input, input.data(), sizeof(float)*128*1*128*128, cudaMemcpyHostToDevice) );
    conv.set_weight(filter);

    return 0;    
}