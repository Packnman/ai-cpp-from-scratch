#include <iostream>
#include <stdio.h>
#include "cuda_matrix.h"
#include "tensor.h"
#include "function.h"

// --------------------------
// Function
// --------------------------
Function::Function()
{

}
Function::~Function()
{

}
void Function::backward(
    cuMat& grad,
    std::vector<std::shared_ptr<Tensor>>& inputs,
    std::vector<std::shared_ptr<Tensor>>& outputs
)
{
    throw std::runtime_error(
        "Function::backward is not implemented"
    );
}
std::shared_ptr<Tensor> Function::forward(
    std::vector<std::shared_ptr<Tensor>>& inputs,
    std::vector<std::shared_ptr<Tensor>>& outputs
)
{
    throw std::runtime_error(
        "Function::forward is not implemented"
    );
}
std::shared_ptr<Tensor> Function::operator()(
    std::vector<std::shared_ptr<Tensor>>& inputs
)
{
    std::vector<std::shared_ptr<Tensor>> outputs;

    std::shared_ptr<Tensor> rst =forward( inputs,outputs );
    outputs.push_back( rst );

    Context* lpContext  =new Context(
        // Non
    );
    std::shared_ptr<Context> context    =std::shared_ptr<Context>(lpContext);
    context->_lpFunc    =this;
    context->_spInputs  =inputs;
    for( int i=0;i<outputs.size();i++ )
    {
        context->_wpOutputs.push_back( outputs[i] );
    }

    rst->_spContexts =context;

    return rst;
}

// --------------------------
// Context
// --------------------------
Context::Context()
{
    _lpFunc =nullptr;
}
Context::~Context()
{

}


// --------------------------
// ReLU
// --------------------------
ReLU::ReLU()
{
    // nothing
}
ReLU::~ReLU()
{
    // nothing
}
void ReLU::backward(
    cuMat& grad,
    std::vector<std::shared_ptr<Tensor>>& inputs,
    std::vector<std::shared_ptr<Tensor>>& outputs
)
{
    // 逆伝播
    if( inputs.size()!=1 )
    {
        throw std::runtime_error(
            "ReLU::backward: ReLU requires exactly one input"
        );
    }
    //
    cuda_ReLU_backward(
        inputs[0]->_mGrad,
        inputs[0]->_mData,
        grad
    );
}

std::shared_ptr<Tensor> ReLU::forward(
    std::vector<std::shared_ptr<Tensor>>& inputs,
    std::vector<std::shared_ptr<Tensor>>& outputs
)
{
    // ReLU(x) =max(0,x)
    if( inputs.size()!=1 )
    {
        throw std::runtime_error(
            "ReLU::forward: ReLU requires exactly one input"
        );
    }
    //
    Tensor* lpTensor    =new Tensor(
        inputs[0]->_mData._nRows,
        inputs[0]->_mData._nCols
    );
    std::shared_ptr<Tensor> rst =std::shared_ptr<Tensor>( lpTensor );
    // 順伝播
    cuda_ReLU_forward(
        rst->_mData,
        inputs[0]->_mData
    );

    return rst;
}


// --------------------------
// Linear
// --------------------------
Linear::Linear(Tensor* lpWeight,Tensor* lpBias)
{
    this->_lpmWeight    =lpWeight;
    this->_lpmBias      =lpBias;
}
Linear::~Linear()
{

}
void Linear::backward(
    cuMat& grad,
    std::vector<std::shared_ptr<Tensor>>& inputs,
    std::vector<std::shared_ptr<Tensor>>& outputs
)
{
    // 損失関数 L が各変数に対してどれくらい変化するかを求めている
    // Y = WX + b
    // grad = dL/dY
    // inputs[0]->_mGrad = dL/dX
    // _lpmWeight->_mGrad = dL/dW
    // _lpmBias->_mGrad = dL/db
    //
    // サイズチェック
    if( inputs.size()!=1 )
    {
        throw std::runtime_error(
            "Linear::backward: Linear requires exactly one input"
        );
    }
    // 全要素1行列の作成
    if( (_mTmp._nRows!=1)||(_mTmp._nCols!=inputs[0]->_mData._nCols) )
    {
        _mTmp   =cuMat( 1,inputs[0]->_mData._nCols );
        cuda_fill( _mTmp,1.0f );
    }

    // X.grad += W^T * grad
    cuda_gemm(
        inputs[0]->_mGrad,
        _lpmWeight->_mData,
        grad,
        true,
        false,
        1.0f,
        1.0f
    );

    // W.grad += grad * X^T
    cuda_gemm(
        _lpmWeight->_mGrad,
        grad,
        inputs[0]->_mData,
        false,
        true,
        1.0f,
        1.0f
    );

    // b.grad += grad * ones^T
    cuda_gemm(
        _lpmBias->_mGrad,
        grad,
        _mTmp,
        false,
        true,
        1.0f,
        1.0f
    );
}
std::shared_ptr<Tensor> Linear::forward(
    std::vector<std::shared_ptr<Tensor>>& inputs,
    std::vector<std::shared_ptr<Tensor>>& outputs
)
{
    // Y = WX + b
    if( inputs.size()!=1 )
    {
        throw std::runtime_error(
            "Linear::forward: Linear requires exactly one input"
        );
    }
    // 全要素1行列の作成
    if( (_mTmp._nRows!=1)||(_mTmp._nCols!=inputs[0]->_mData._nCols) )
    {
        _mTmp   =cuMat( 1,inputs[0]->_mData._nCols );
        cuda_fill( _mTmp,1.0f );
    }
    //
    Tensor* lpTensor    =new Tensor(
        _lpmWeight->_mData._nRows,
        inputs[0]->_mData._nCols
    );
    std::shared_ptr<Tensor> rst =std::shared_ptr<Tensor>( lpTensor );
    
    // u = W * X
    cuda_gemm(
        rst->_mData,
        _lpmWeight->_mData,
        inputs[0]->_mData
    );
    // biasを各batchに加える
    // _mTmp = [1 1 1 ... 1]
    // u += b * _mTmp
    cuda_gemm(
        rst->_mData,
        _lpmBias->_mData,
        _mTmp,
        false,
        false,
        1.0f,
        1.0f
    );

    return rst;
}

// --------------------------
// GELU
// --------------------------
GELU::GELU()
{

}
GELU::~GELU()
{

}
void GELU::backward(
    cuMat& grad,
    std::vector<std::shared_ptr<Tensor>>& inputs,
    std::vector<std::shared_ptr<Tensor>>& outputs
)
{
    // 内容はREADMEを参照すること
    if( inputs.size()!=1 )
    {
        throw std::runtime_error(
            "GELU::backward: GELU requires exactly one input"
        );
    }
    //
    cuda_GELU_backward(
        inputs[0]->_mGrad,
        inputs[0]->_mData,
        grad
    );
}
std::shared_ptr<Tensor> GELU::forward(
    std::vector<std::shared_ptr<Tensor>>& inputs,
    std::vector<std::shared_ptr<Tensor>>& outputs
)
{
    // 内容はREADMEを参照すること
    if( inputs.size()!=1 )
    {
        throw std::runtime_error(
            "GELU::forward: GELU requires exactly one input"
        );
    }

    Tensor* lpTensor    =new Tensor(
        inputs[0]->_mData._nRows,
        inputs[0]->_mData._nCols
    );
    std::shared_ptr<Tensor> rst =std::shared_ptr<Tensor>( lpTensor );
    //
    cuda_GELU_forward(
        rst->_mData,
        inputs[0]->_mData
    );

    return rst;
}

// --------------------------
// SoftmaxCrossEntropy
// --------------------------
SoftmaxCrossEntropy::SoftmaxCrossEntropy()
{

}
SoftmaxCrossEntropy::~SoftmaxCrossEntropy()
{

}
void SoftmaxCrossEntropy::backward(
    cuMat& grad,
    std::vector<std::shared_ptr<Tensor>>& inputs,
    std::vector<std::shared_ptr<Tensor>>& outputs
)
{
    // inputs[0] = logits
    // inputs[1] = target(one-hot)
    //
    // L = SoftmaxCrossEntropy(logits, target)
    //
    // dL/dlogits
    //     = (softmax(logits) - target) / batch_size
    //
    // logits.grad += grad * dL/dlogits
    // 
    if( inputs.size()!=2 )
    {
        throw std::runtime_error(
            "SoftmaxCrossEntropy::backward: "
            "SoftmaxCrossEntropy requires exactly two inputs"
        );
    }
    //
    cuda_SoftmaxCrossEntropy_backward(
        inputs[0]->_mGrad,
        inputs[0]->_mData,
        inputs[1]->_mData,
        grad
    );
}
std::shared_ptr<Tensor> SoftmaxCrossEntropy::forward(
    std::vector<std::shared_ptr<Tensor>>& inputs,
    std::vector<std::shared_ptr<Tensor>>& outputs
)
{
    // inputs[0] = logits
    // inputs[1] = target(one-hot)
    if( inputs.size()!=2 )
    {
        throw std::runtime_error(
            "SoftmaxCrossEntropy::forward: "
            "SoftmaxCrossEntropy requires exactly two inputs"
        );
    }
    if( (inputs[0]->_mData._nRows!=inputs[1]->_mData._nRows)||
        (inputs[0]->_mData._nCols!=inputs[1]->_mData._nCols) )
    {
        throw std::runtime_error(
            "SoftmaxCrossEntropy::forward: "
            "logits and target size mismatch"
        );
    }
    //
    Tensor* lpTensor    =new Tensor(
        1,
        1
    );
    std::shared_ptr<Tensor> rst =std::shared_ptr<Tensor>( lpTensor );

    cuda_SoftmaxCrossEntropy_forward(
        rst->_mData,
        inputs[0]->_mData,
        inputs[1]->_mData
    );

    return rst;
}