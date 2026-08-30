#pragma once

#include <cstdint>
#include <curand.h>
#include <vector>
#include <memory>
#include "cuda_matrix.h"

class Tensor;

using TensorPtr =std::shared_ptr<Tensor>;
using TensorList =std::vector<TensorPtr>;
using TensorGradList =std::vector<const cuMat*>;

// --------------------------
// Function
// --------------------------
class Function{
public:
    Function();
    virtual ~Function();

public:
    virtual void backward(
        const TensorGradList& c_lpmOutputGrads,
        const TensorList& c_spmInputs,
        const TensorList& c_spmOutputs
    ) =0;
    virtual TensorList forward(const TensorList& c_spmInputs) =0;

    TensorList apply(const TensorList& c_spmInputs);
    TensorPtr operator()(const TensorList& c_spmInputs);
};

// --------------------------
// Context
// --------------------------
class Context{
public:
    Context();
    virtual ~Context();

public:
    Function* _lpFunc;

    TensorList _spmInputs;
    std::vector<std::weak_ptr<Tensor>> _wpmOutputs;
};

// --------------------------
// ReLU
// --------------------------
class ReLU: public Function
{
public:
    ReLU();
    ~ReLU();

public:
    void backward(
        const TensorGradList& c_lpmOutputGrads,
        const TensorList& c_spmInputs,
        const TensorList& c_spmOutputs
    ) override;
    TensorList forward(const TensorList& c_spmInputs) override;
};

// --------------------------
// Dropout
// --------------------------
class Dropout: public Function
{
public:
    Dropout(float fDropProbability,std::uint64_t nSeed);
    ~Dropout() override;

private:
    float _fDropProbability;
    curandGenerator_t _crnGenerator;
    cuMat _mMask;

public:
    void backward(
        const TensorGradList& c_lpmOutputGrads,
        const TensorList& c_spmInputs,
        const TensorList& c_spmOutputs
    ) override;
    TensorList forward(const TensorList& c_spmInputs) override;
};

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
        const TensorGradList& c_lpmOutputGrads,
        const TensorList& c_spmInputs,
        const TensorList& c_spmOutputs
    ) override;
    TensorList forward(const TensorList& c_spmInputs) override;
};

// --------------------------
// Linear
// --------------------------
class Linear: public Function
{
public:
    Linear(Tensor* lpWeight,Tensor* lpBias);
    ~Linear();

public:
    Tensor* _lpmWeight;
    Tensor* _lpmBias;
    cuMat   _mTmp;
public:
    void backward(
        const TensorGradList& c_lpmOutputGrads,
        const TensorList& c_spmInputs,
        const TensorList& c_spmOutputs
    ) override;
    TensorList forward(const TensorList& c_spmInputs) override;
};

// --------------------------
// GELU
// --------------------------
class GELU: public Function
{
public:
    GELU();
    ~GELU();

public:
    void backward(
        const TensorGradList& c_lpmOutputGrads,
        const TensorList& c_spmInputs,
        const TensorList& c_spmOutputs
    ) override;
    TensorList forward(const TensorList& c_spmInputs) override;
};

// --------------------------
// SoftmaxCrossEntropy
// --------------------------
class SoftmaxCrossEntropy: public Function
{
public:
    SoftmaxCrossEntropy();
    ~SoftmaxCrossEntropy();

public:
    void backward(
        const TensorGradList& c_lpmOutputGrads,
        const TensorList& c_spmInputs,
        const TensorList& c_spmOutputs
    ) override;
    TensorList forward(const TensorList& c_spmInputs) override;
};
