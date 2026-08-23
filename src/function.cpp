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
    // nothing
}
Function::~Function()
{
    // nothing
}
void Function::backward(
    cuMat& grad,
    std::vector<std::shared_ptr<Tensor>>& inputs,
    std::vector<std::shared_ptr<Tensor>>& outputs
)
{
    // 逆伝播
}
std::shared_ptr<Tensor> Function::forward(
    std::vector<std::shared_ptr<Tensor>>& inputs,
    std::vector<std::shared_ptr<Tensor>>& outputs
)
{
    // 順伝播
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
    // 順伝播
    Tensor* lpTensor    =new Tensor(
        this,
        inputs[0]->_mData._nRows,
        inputs[0]->data().get_nCols()
    );
    std::shared_ptr<Tensor> rst =std::shared_ptr<Tensor>( lpTensor );

    // TODO

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
    std::shared_ptr<Tensor> x   =inputs[0];

    x->grad()           +=_lpmWeight->data().transpose() * grad;
    _lpmWeight->grad()  +=grad * x->data().transpose();
    _lpmBias->grad()    +=grad * _mTmp.transpose();
}
std::shared_ptr<Tensor> Linear::forward(
    std::vector<std::shared_ptr<Tensor>>& inputs,
    std::vector<std::shared_ptr<Tensor>>& outputs
)
{
    std::shared_ptr<Tensor> x   =inputs[0];

    Tensor* lpTensor    =new Tensor(
        this,
        _lpmWeight->data().get_nRows(),
        x->data().get_nCols()
    );
    std::shared_ptr<Tensor> u   =std::shared_ptr<Tensor>( lpTensor );
    // 全要素1行列の作成
    if( (_mTmp.get_nCols()==0)||(_mTmp.get_nCols()!=x->data().get_nCols()) )
    {
        _mTmp   =cuMat( 1,x->data().get_nCols() );
        _mTmp.ones();
    }

    u->data()   =_lpmWeight->data() * x->data() + _lpmBias->data() * _mTmp;

    return u;
}
