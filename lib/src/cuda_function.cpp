#include <iostream>
#include <stdio.h>
#include <string>
#include "cuda_matrix.h"
#include "cuda_tensor.h"
#include "cuda_function.h"

namespace {
    const cuMat& requireSingleOutputGrad(
        const TensorGradList& c_lpmOutputGrads,
        const char* c_lpszFunctionName
    )
    {
        if( (c_lpmOutputGrads.size()!=1)||(c_lpmOutputGrads[0]==nullptr) )
        {
            throw std::runtime_error(
                std::string(c_lpszFunctionName)+": exactly one output gradient is required"
            );
        }
        return *c_lpmOutputGrads[0];
    }
}

// --------------------------
// Function
// --------------------------
Function::Function()
{

}
Function::~Function()
{

}
TensorList Function::apply(const TensorList& c_spmInputs)
{
    TensorList spmOutputs =forward( c_spmInputs );
    if( spmOutputs.empty() )
    {
        throw std::runtime_error(
            "Function::apply: forward must return at least one output"
        );
    }

    auto spContext  =std::make_shared<Context>();
    spContext->_lpFunc      =this;
    spContext->_spmInputs   =c_spmInputs;
    for( const auto& c_spmOutput : spmOutputs )
    {
        if( c_spmOutput==nullptr )
        {
            throw std::runtime_error(
                "Function::apply: forward returned a null output"
            );
        }
        spContext->_wpmOutputs.push_back( c_spmOutput );
        c_spmOutput->_spContext =spContext;
    }

    return spmOutputs;
}

TensorPtr Function::operator()(const TensorList& c_spmInputs)
{
    TensorList spmOutputs =apply( c_spmInputs );
    if( spmOutputs.size()!=1 )
    {
        throw std::runtime_error(
            "Function::operator(): use apply() for a multi-output Function"
        );
    }

    return spmOutputs[0];
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
    const TensorGradList& c_lpmOutputGrads,
    const TensorList& c_spmInputs,
    const TensorList& c_spmOutputs
)
{
    (void)c_spmOutputs;
    // 逆伝播
    if( c_spmInputs.size()!=1 )
    {
        throw std::runtime_error(
            "ReLU::backward: ReLU requires exactly one input"
        );
    }
    //
    cuda_ReLU_backward(
        c_spmInputs[0]->_mGrad,
        c_spmInputs[0]->_mData,
        requireSingleOutputGrad(c_lpmOutputGrads,"ReLU::backward")
    );
}

TensorList ReLU::forward(const TensorList& c_spmInputs)
{
    // ReLU(x) =max(0,x)
    if( c_spmInputs.size()!=1 )
    {
        throw std::runtime_error(
            "ReLU::forward: ReLU requires exactly one input"
        );
    }
    //
    auto spmResult =std::make_shared<Tensor>(
        c_spmInputs[0]->_mData._nRows,
        c_spmInputs[0]->_mData._nCols
    );
    // 順伝播
    cuda_ReLU_forward(
        spmResult->_mData,
        c_spmInputs[0]->_mData
    );

    return {spmResult};
}


// --------------------------
// Linear
// --------------------------
Linear::Linear(Tensor* lpWeight,Tensor* lpBias)
    :_lpmWeight( lpWeight ),
     _lpmBias( lpBias )
{
    
}
Linear::~Linear()
{

}
void Linear::backward(
    const TensorGradList& c_lpmOutputGrads,
    const TensorList& c_spmInputs,
    const TensorList& c_spmOutputs
)
{
    (void)c_spmOutputs;
    // 損失関数 L が各変数に対してどれくらい変化するかを求めている
    // Y = WX + b
    // grad = dL/dY
    // c_spmInputs[0]->_mGrad = dL/dX
    // _lpmWeight->_mGrad = dL/dW
    // _lpmBias->_mGrad = dL/db
    //
    // サイズチェック
    if( c_spmInputs.size()!=1 )
    {
        throw std::runtime_error(
            "Linear::backward: Linear requires exactly one input"
        );
    }
    const cuMat& c_mGrad =requireSingleOutputGrad(
        c_lpmOutputGrads,
        "Linear::backward"
    );
    // 全要素1行列の作成
    if( (_mTmp._nRows!=1)||(_mTmp._nCols!=c_spmInputs[0]->_mData._nCols) )
    {
        _mTmp   =cuMat( 1,c_spmInputs[0]->_mData._nCols );
        cuda_fill( _mTmp,1.0f );
    }

    // X.grad += W^T * grad
    cuda_gemm(
        c_spmInputs[0]->_mGrad,
        _lpmWeight->_mData,
        c_mGrad,
        true,
        false,
        1.0f,
        1.0f
    );

    // W.grad += grad * X^T
    cuda_gemm(
        _lpmWeight->_mGrad,
        c_mGrad,
        c_spmInputs[0]->_mData,
        false,
        true,
        1.0f,
        1.0f
    );

    // b.grad += grad * ones^T
    cuda_gemm(
        _lpmBias->_mGrad,
        c_mGrad,
        _mTmp,
        false,
        true,
        1.0f,
        1.0f
    );
}
TensorList Linear::forward(const TensorList& c_spmInputs)
{
    // Y = WX + b
    if( c_spmInputs.size()!=1 )
    {
        throw std::runtime_error(
            "Linear::forward: Linear requires exactly one input"
        );
    }
    // 全要素1行列の作成
    if( (_mTmp._nRows!=1)||(_mTmp._nCols!=c_spmInputs[0]->_mData._nCols) )
    {
        _mTmp   =cuMat( 1,c_spmInputs[0]->_mData._nCols );
        cuda_fill( _mTmp,1.0f );
    }
    //
    auto spmResult =std::make_shared<Tensor>(
        _lpmWeight->_mData._nRows,
        c_spmInputs[0]->_mData._nCols
    );
    
    // u = W * X
    cuda_gemm(
        spmResult->_mData,
        _lpmWeight->_mData,
        c_spmInputs[0]->_mData
    );
    // biasを各batchに加える
    // _mTmp = [1 1 1 ... 1]
    // u += b * _mTmp
    cuda_gemm(
        spmResult->_mData,
        _lpmBias->_mData,
        _mTmp,
        false,
        false,
        1.0f,
        1.0f
    );

    return {spmResult};
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
    const TensorGradList& c_lpmOutputGrads,
    const TensorList& c_spmInputs,
    const TensorList& c_spmOutputs
)
{
    (void)c_spmOutputs;
    // 内容はREADMEを参照すること
    if( c_spmInputs.size()!=1 )
    {
        throw std::runtime_error(
            "GELU::backward: GELU requires exactly one input"
        );
    }
    //
    cuda_GELU_backward(
        c_spmInputs[0]->_mGrad,
        c_spmInputs[0]->_mData,
        requireSingleOutputGrad(c_lpmOutputGrads,"GELU::backward")
    );
}
TensorList GELU::forward(const TensorList& c_spmInputs)
{
    // 内容はREADMEを参照すること
    if( c_spmInputs.size()!=1 )
    {
        throw std::runtime_error(
            "GELU::forward: GELU requires exactly one input"
        );
    }

    auto spmResult =std::make_shared<Tensor>(
        c_spmInputs[0]->_mData._nRows,
        c_spmInputs[0]->_mData._nCols
    );
    //
    cuda_GELU_forward(
        spmResult->_mData,
        c_spmInputs[0]->_mData
    );

    return {spmResult};
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
    const TensorGradList& c_lpmOutputGrads,
    const TensorList& c_spmInputs,
    const TensorList& c_spmOutputs
)
{
    (void)c_spmOutputs;
    // c_spmInputs[0] = logits
    // c_spmInputs[1] = target(one-hot)
    //
    // L = SoftmaxCrossEntropy(logits, target)
    //
    // dL/dlogits
    //     = (softmax(logits) - target) / batch_size
    //
    // logits.grad += grad * dL/dlogits
    // 
    if( c_spmInputs.size()!=2 )
    {
        throw std::runtime_error(
            "SoftmaxCrossEntropy::backward: "
            "SoftmaxCrossEntropy requires exactly two inputs"
        );
    }
    //
    cuda_SoftmaxCrossEntropy_backward(
        c_spmInputs[0]->_mGrad,
        c_spmInputs[0]->_mData,
        c_spmInputs[1]->_mData,
        requireSingleOutputGrad(c_lpmOutputGrads,"SoftmaxCrossEntropy::backward")
    );
}
TensorList SoftmaxCrossEntropy::forward(const TensorList& c_spmInputs)
{
    // c_spmInputs[0] = logits
    // c_spmInputs[1] = target(one-hot)
    if( c_spmInputs.size()!=2 )
    {
        throw std::runtime_error(
            "SoftmaxCrossEntropy::forward: "
            "SoftmaxCrossEntropy requires exactly two inputs"
        );
    }
    if( (c_spmInputs[0]->_mData._nRows!=c_spmInputs[1]->_mData._nRows)||
        (c_spmInputs[0]->_mData._nCols!=c_spmInputs[1]->_mData._nCols) )
    {
        throw std::runtime_error(
            "SoftmaxCrossEntropy::forward: "
            "logits and target size mismatch"
        );
    }
    //
    auto spmResult =std::make_shared<Tensor>( 1,1 );

    cuda_SoftmaxCrossEntropy_forward(
        spmResult->_mData,
        c_spmInputs[0]->_mData,
        c_spmInputs[1]->_mData
    );

    return {spmResult};
}
