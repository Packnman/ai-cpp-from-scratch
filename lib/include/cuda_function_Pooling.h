#pragma once

#include "cuda_function.h"



// --------------------------
// Pooling
// --------------------------
class Pooling : public Function
{
public:
    Pooling(
        int nChannels,
        int nInputHeight,
        int nInputWidth,
        int nKernelSize =2,
        int nStride =2
    );
    ~Pooling() override;
private:
    int _nChannels;
    int _nInputHeight;
    int _nInputWidth;
    int _nKernelSize;
    int _nStride;
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
