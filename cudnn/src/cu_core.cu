#include <jhdnn.cuh>


cudnnHandle_t* cudnn;

/***************************************************************
 * Debug code
 ***************************************************************/
void cudnnAssert(cudnnStatus_t code, const char *file, int line) {
    if (code != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr,"cuDNN assert: %s %s %d\n", cudnnGetErrorString(code), file, line);
        exit(1);
    }
}

void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(1);
    }
}

void cudnn_create() {

    /******************************************************************
     * 1. Define cudnn Handler
     *******************************************************************/
    cudnn = new cudnnHandle_t;
    cudnnErrChk( cudnnCreate(cudnn) );
};

void cudnn_destroy() {
    cudnnErrChk( cudnnDestroy(*cudnn) );
    delete cudnn ;
}
