set(APP conv_forward.out)

set(MAIN conv.cu)
set(SRC_FILES 
)

add_executable(${APP} ${MAIN} ${SRC_FILES})
target_include_directories(${APP} PUBLIC ${PATH_JHDNN_INC} ${PATH_CUDNN_INC})

target_link_libraries(${APP} PUBLIC ${LIB_NAME} cudnn)