#pragma once

#include "cuda_function.h"


// --------------------------
// LayerNorm
// --------------------------
class LayerNorm: public Function
{
public:
    LayerNorm();
    LayerNorm(Tensor* lpGamma,Tensor* lpBeta,float fEpsilon =1.0e-5f);
    ~LayerNorm();

private:
    struct ShapeInfo
    {
        int nFeatures;
        int nPositions;
    };

    Tensor* _lpmGamma;
    Tensor* _lpmBeta;
    float _fEpsilon;

    static void _validateParameter(
        const Tensor* c_lpParameter,
        std::int64_t c_nFeatures,
        const char* c_lpszName
    );
    ShapeInfo _validateInput(
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
