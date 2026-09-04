#pragma once

#include "cuda_function.h"


// --------------------------
// Conv2D
// --------------------------
class Conv2D : public Function
{
public:
    Conv2D(
        Tensor* lpWeight,
        Tensor* lpBias,
        int nInputChannels,
        int nInputHeight,
        int nInputWidth,
        int nKernelSize,
        int nStride =1,
        int nPadding =0
    );
    ~Conv2D() override;
private:
    Tensor* _lpmWeight;
    Tensor* _lpmBias;
    int _nInputChannels;
    int _nInputHeight;
    int _nInputWidth;
    int _nKernelSize;
    int _nStride;
    int _nPadding;
    int _nOutputChannels;
    int _nOutputHeight;
    int _nOutputWidth;
public:
    void backward(
        const std::vector<const cuMat*>& c_lpmOutputGrads,
        const std::vector<std::shared_ptr<Tensor>>& c_spmInputs,
        const std::vector<std::shared_ptr<Tensor>>& c_spmOutputs
    ) override;
    std::vector<std::shared_ptr<Tensor>> forward(
        const std::vector<std::shared_ptr<Tensor>>& c_spmInputs
    ) override;
};