#pragma once


/***************************************************************
 * Layer
 ***************************************************************/
 class jhLayerFloat {

protected:

    /** Device memory **/
    float* d_dx;
    float* d_dy;
    float* d_x;
    float* d_y;

public:
    jhLayerFloat(){};
    ~jhLayerFloat(){};
    virtual void forward(float* x){};
    virtual void backward(float* dy){};
    float* get_y() {return d_y;};
    float* get_dx() {return d_dx;};
};