#include "cuda_tensor.h"
#include "cuda_function_Linear.h"


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
    const std::vector<const cuMat*>& c_lpmOutputGrads,
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs,
    const std::vector<std::shared_ptr<Tensor>>& c_spmOutputs
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
std::vector<std::shared_ptr<Tensor>>
Linear::forward(
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs
)
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
