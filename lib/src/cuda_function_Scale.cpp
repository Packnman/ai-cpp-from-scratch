#include "cuda_function_Scale.h"

#include <climits>
#include <cmath>
#include <stdexcept>

#include "cuda_tensor.h"

Scale::Scale(float fScale)
    :_fScale( fScale )
{
    if( !std::isfinite(_fScale) )
    {
        throw std::invalid_argument("Scale: scale must be finite");
    }
}

Scale::~Scale()
{
}

void Scale::backward(
    const TensorGradList& c_lpmOutputGrads,
    const TensorList& c_spmInputs,
    const TensorList& c_spmOutputs
)
{
    (void)c_spmOutputs;
    if( (c_spmInputs.size()!=1)||(c_spmInputs[0]==nullptr) )
    {
        throw std::runtime_error("Scale::backward: exactly one input is required");
    }
    const cufMat& c_mGrad =requireSingleOutputGrad(
        c_lpmOutputGrads,"Scale::backward"
    );
    if( c_mGrad.shape()!=c_spmInputs[0]->_mData.shape() )
    {
        throw std::invalid_argument("Scale::backward: gradient shape mismatch");
    }
    if( !c_mGrad.isContiguous()||
        !c_spmInputs[0]->_mGrad.isContiguous() )
    {
        throw std::invalid_argument(
            "Scale::backward: contiguous tensors required"
        );
    }
    if( c_mGrad.numel()>static_cast<std::size_t>(INT_MAX) )
    {
        throw std::overflow_error("Scale::backward: tensor is too large");
    }
    cuda_axpy( c_spmInputs[0]->_mGrad,_fScale,c_mGrad );
}

TensorList Scale::forward(const TensorList& c_spmInputs)
{
    if( (c_spmInputs.size()!=1)||(c_spmInputs[0]==nullptr) )
    {
        throw std::runtime_error("Scale::forward: exactly one input is required");
    }
    if( !c_spmInputs[0]->_mData.isContiguous() )
    {
        throw std::invalid_argument(
            "Scale::forward: contiguous input required"
        );
    }
    if( c_spmInputs[0]->_mData.numel()>
        static_cast<std::size_t>(INT_MAX) )
    {
        throw std::overflow_error("Scale::forward: tensor is too large");
    }
    auto spmResult =std::make_shared<Tensor>(
        c_spmInputs[0]->_mData.shape()
    );
    cuda_geam(
        spmResult->_mData,_fScale,c_spmInputs[0]->_mData,
        0.0f,c_spmInputs[0]->_mData
    );
    return {spmResult};
}
