#pragma once

#include "cuda_function.h"

// --------------------------
// Softmax
// --------------------------
class Softmax: public Function{
public:
    explicit Softmax(std::size_t nAxis =0);
    ~Softmax();

private:
    struct ShapeInfo
    {
        int _nOuter;
        int _nAxisSize;
        int _nInner;
        int _nSlices;
    };

    std::size_t _nAxis;

    ShapeInfo _validateInput(
        const cufMat& c_mInput,
        const char* c_lpszOperation
    ) const;

public:
    void backward(
        const std::vector<const cufMat*>& c_lpmOutputGrads,
        const std::vector<std::shared_ptr<Tensor>>& c_spmInputs,
        const std::vector<std::shared_ptr<Tensor>>& c_spmOutputs
    ) override;
    std::vector<std::shared_ptr<Tensor>> forward(
        const std::vector<std::shared_ptr<Tensor>>& c_spmInputs
    ) override;
};
