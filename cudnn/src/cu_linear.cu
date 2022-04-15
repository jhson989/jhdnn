#include <jhdnn.cuh>

cuLinearFloat::cuLinearFloat(
    const int BATCH_NUM_,
    const int IN_FEATURES_,
    const int OUT_FEATURES_
) : BATCH_NUM(BATCH_NUM_), IN_FEATURES(IN_FEATURES_), OUT_FEATURES(OUT_FEATURES_) {

        
    /***************************************************************
     * 1. Tensor descriptor : A, B, C (C = A*B)
     ***************************************************************/
    
    // Tensor A
    cudnnErrChk( cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &desc_A) );




}

cuLinearFloat::~cuLinearFloat() {

    cudnnErrChk( cudnnBackendDestroyDescriptor(desc_A) );

}