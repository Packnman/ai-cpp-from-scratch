#pragma once

#include "cuda_function.h"


// --------------------------
// BatchNorm
// --------------------------
class BatchNorm: public Function
{
public:
    BatchNorm(
        Tensor* lpGamma,
        Tensor* lpBeta,
        Tensor* lpRunningMean,
        Tensor* lpRunningVar,
        float fMomentum=0.1f,
        float fEpsilon=1.0e-5f
    );
    ~BatchNorm() override;

private:
    Tensor* _lpmGamma;
    Tensor* _lpmBeta;
    Tensor* _lpmRunningMean;
    Tensor* _lpmRunningVar;
    float   _fMomentum;
    float   _fEpsilon;
    bool    _isTraining;
    bool    _wasTraining;
    cuMat   _mNormalized;
    cuMat   _mInvStd;

public:
    void setTraining(bool isTraining);
    bool isTraining() const;

    void backward(
        const std::vector<const cuMat*>& c_lpmOutputGrads,
        const std::vector<std::shared_ptr<Tensor>>& c_spmInputs,
        const std::vector<std::shared_ptr<Tensor>>& c_spmOutputs
    ) override;
    std::vector<std::shared_ptr<Tensor>> forward(
        const std::vector<std::shared_ptr<Tensor>>& c_spmInputs
    ) override;
};
