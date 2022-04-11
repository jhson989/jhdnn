# CMake bianry version
cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_COMPILER /usr/local/cuda/bin/nvcc)

# Proejct info
project(
    jhDNN
    VERSION 0.1
    DESCRIPTION "jhDNN Implementation"
    LANGUAGES CXX CUDA
)

set(LIB_NAME jhdnn)

# Global include
set(PATH_INC ./include)
set(INC_FILES 
    "./include/jhdnn.cuh\;cudnn/include/cu_core.cuh\;cudnn/include/cu_conv.cuh\;cudnn/include/cu_dense.cuh"
)

# cudnn include
set(PATH_CUDNN_INC ./cudnn/include)
set(SRC_CUDNN_FILES
    cudnn/src/cu_core.cu
    cudnn/src/cu_conv.cu
    cudnn/src/cu_dense.cu
)

# jhdnn include
set(PATH_JHDNN_INC ./jhdnn/include)
set(SRC_JHDNN_FILES
)

add_library(${LIB_NAME} SHARED ${SRC_CUDNN_FILES})
target_include_directories(
    ${LIB_NAME} 
    PUBLIC ${PATH_INC} ${PATH_CUDNN_INC} ${PATH_JHDNN_INC}
)
target_link_libraries(${LIB_NAME} cudnn)
set_target_properties(
    ${LIB_NAME}
    PROPERTIES PUBLIC_HEADER ${INC_FILES}
)


# Install
set(PATH_LIB_INSTALL /home/jhson/.lib/)
set(PATH_INC_INSTALL /home/jhson/.include/)

install(
    TARGETS ${LIB_NAME}
    LIBRARY DESTINATION ${PATH_LIB_INSTALL}
    PUBLIC_HEADER DESTINATION ${PATH_INC_INSTALL}
)

add_subdirectory(test)
