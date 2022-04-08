

#include <jhdnn.cuh>

int main(void) {

    cuConvFloat conv(
        128, 3, 128, 128,
        3, 128, 128,
        3, 3
    );


    return 0;
    
}