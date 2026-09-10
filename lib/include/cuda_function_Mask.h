#pragma once

#include "cuda_function.h"


// --------------------------
// Mask
// --------------------------
class Mask: public Function
{
public:
    explicit Mask(Tensor* lpMask);
    ~Mask();

private:
    struct ShapeInfo
    {
        int _nQuery;
        int _nKey;
        int _nTrailing;
        bool _isBroadcast;
    };

    Tensor* _lpmMask;

    ShapeInfo validateInput(
        const std::vector<std::shared_ptr<Tensor>>& c_spmInputs,
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
