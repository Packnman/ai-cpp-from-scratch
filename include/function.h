#pragma once

#include <vector>
#include <memory>
#include "cuMat.h"

class Tensor;

// --------------------------
// Function
// --------------------------
class Function{
public:
    Function();
    virtual ~Function();
private:
protected:
    std::vector<std::shared_ptr<Tensor>>  _spInputs;
    std::vector<std::shared_ptr<Tensor>>  _spOutputs;
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
// GeRU
// --------------------------
class GeRU: public Function
{
public:
    GeRU();
    ~GeRU();
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