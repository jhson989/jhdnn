#include <jhdnn.cuh>


cudnnHandle_t* cudnn;

/***************************************************************
 * Debug code
 ***************************************************************/
void cudnnAssert(cudnnStatus_t code, const char *file, int line) {
   if (code != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr,"cuDNN assert: %s %s %d\n", cudnnGetErrorString(code), file, line);
   }
}

void cudaAssert(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      fprintf(stderr,"CUDA assert: %s %s %d\n", cudaGetErrorString(code), file, line);
   }
}
