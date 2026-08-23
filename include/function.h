#pragma once

#include <vector>
#include <memory>
#include "cuda_matrix.h"

class Tensor;

// --------------------------
// Function
// --------------------------
class Function{
public:
    Function();
    virtual ~Function();
private:
public:
    virtual void backward(
        cuMat& grad,
        std::vector<std::shared_ptr<Tensor>>& inputs,
        std::vector<std::shared_ptr<Tensor>>& outputs
    );
    virtual std::shared_ptr<Tensor> forward(
        std::vector<std::shared_ptr<Tensor>>& inputs,
        std::vector<std::shared_ptr<Tensor>>& outputs
    );
    //
    std::shared_ptr<Tensor> operator()(
        std::vector<std::shared_ptr<Tensor>>& inputs
    );
};

// --------------------------
// Context
// --------------------------
class Context{
public:
    Context();
    virtual ~Context();
private:
protected:
public:
    Function* _lpFunc;

    std::vector<std::shared_ptr<Tensor>> _spInputs;
    std::vector<std::weak_ptr<Tensor>>   _wpOutputs;
};

// --------------------------
// ReLU
// --------------------------
class ReLU: public Function
{
public:
    ReLU();
    ~ReLU();
protected:

public:
    virtual void backward(
        cuMat& grad,
        std::vector<std::shared_ptr<Tensor>>& inputs,
        std::vector<std::shared_ptr<Tensor>>& outputs
    );
    virtual std::shared_ptr<Tensor> forward(
        std::vector<std::shared_ptr<Tensor>>& inputs,
        std::vector<std::shared_ptr<Tensor>>& outputs
    );
};

// --------------------------
// Linear
// --------------------------
class Linear: public Function
{
public:
    Linear(Tensor* lpWeight,Tensor* lpBias);
    ~Linear();
protected:
public:
    Tensor* _lpmWeight;
    Tensor* _lpmBias;
    cuMat   _mTmp;
public:
    virtual void backward(
        cuMat& grad,
        std::vector<std::shared_ptr<Tensor>>& inputs,
        std::vector<std::shared_ptr<Tensor>>& outputs
    );
    virtual std::shared_ptr<Tensor> forward(
        std::vector<std::shared_ptr<Tensor>>& inputs,
        std::vector<std::shared_ptr<Tensor>>& outputs
    );
};

// --------------------------
// GELU
// --------------------------
class GELU: public Function
{
public:
    GELU();
    ~GELU();
protected:

public:
    virtual void backward(
        cuMat& grad,
        std::vector<std::shared_ptr<Tensor>>& inputs,
        std::vector<std::shared_ptr<Tensor>>& outputs
    );
    virtual std::shared_ptr<Tensor> forward(
        std::vector<std::shared_ptr<Tensor>>& inputs,
        std::vector<std::shared_ptr<Tensor>>& outputs
    );
};

// --------------------------
// SoftmaxCrossEntropy
// --------------------------
class SoftmaxCrossEntropy: public Function
{
public:
    SoftmaxCrossEntropy();
    ~SoftmaxCrossEntropy();
protected:

public:
    virtual void backward(
        cuMat& grad,
        std::vector<std::shared_ptr<Tensor>>& inputs,
        std::vector<std::shared_ptr<Tensor>>& outputs
    );
    virtual std::shared_ptr<Tensor> forward(
        std::vector<std::shared_ptr<Tensor>>& inputs,
        std::vector<std::shared_ptr<Tensor>>& outputs
    );
};