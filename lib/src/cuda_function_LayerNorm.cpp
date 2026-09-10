#include "cuda_function_LayerNorm.h"

#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_tensor.h"


void LayerNorm::_validateParameter(
    const Tensor* c_lpParameter,
    std::int64_t c_nFeatures,
    const char* c_lpszName
)
{
    if( c_lpParameter==nullptr )
    {
        throw std::invalid_argument(std::string("LayerNorm: null ")+c_lpszName);
    }
    if( c_lpParameter->_mData.shape()!=
        std::vector<std::int64_t>{c_nFeatures,1} )
    {
        throw std::invalid_argument(
            std::string("LayerNorm: invalid ")+c_lpszName+" shape"
        );
    }
    if( !c_lpParameter->_mData.isContiguous()||
        !c_lpParameter->_mGrad.isContiguous() )
    {
        throw std::invalid_argument(
            std::string("LayerNorm: non-contiguous ")+c_lpszName
        );
    }
}

LayerNorm::LayerNorm()
    :_lpmGamma( nullptr ),
     _lpmBeta( nullptr ),
     _fEpsilon( 1.0e-5f )
{
}

LayerNorm::LayerNorm(Tensor* lpGamma,Tensor* lpBeta,float fEpsilon)
    :_lpmGamma( lpGamma ),
     _lpmBeta( lpBeta ),
     _fEpsilon( fEpsilon )
{
    if( !std::isfinite(_fEpsilon)||(_fEpsilon<=0.0f) )
    {
        throw std::invalid_argument(
            "LayerNorm: epsilon must be finite and positive"
        );
    }
    if( (_lpmGamma==nullptr)!=(_lpmBeta==nullptr) )
    {
        throw std::invalid_argument(
            "LayerNorm: gamma and beta must both be provided"
        );
    }
    if( _lpmGamma!=nullptr )
    {
        if( (_lpmGamma->_mData.dim()!=2)||
            (_lpmGamma->_mData.size(0)<=0) )
        {
            throw std::invalid_argument("LayerNorm: invalid gamma shape");
        }
        _validateParameter( _lpmGamma,_lpmGamma->_mData.size(0),"gamma" );
        _validateParameter( _lpmBeta,_lpmGamma->_mData.size(0),"beta" );
    }
}

LayerNorm::~LayerNorm()
{
}

LayerNorm::ShapeInfo LayerNorm::_validateInput(
    const TensorList& c_spmInputs,
    const char* c_lpszOperation
) const
{
    if( (c_spmInputs.size()!=1)||(c_spmInputs[0]==nullptr) )
    {
        throw std::runtime_error(
            std::string(c_lpszOperation)+": exactly one input is required"
        );
    }
    if( !std::isfinite(_fEpsilon)||(_fEpsilon<=0.0f) )
    {
        throw std::invalid_argument(
            std::string(c_lpszOperation)+
            ": epsilon must be finite and positive"
        );
    }

    const cufMat& c_mInput =c_spmInputs[0]->_mData;
    if( (c_mInput.dim()<1)||(c_mInput.size(0)<=0) )
    {
        throw std::invalid_argument(
            std::string(c_lpszOperation)+": invalid input shape"
        );
    }
    if( !c_mInput.isContiguous()||
        !c_spmInputs[0]->_mGrad.isContiguous() )
    {
        throw std::invalid_argument(
            std::string(c_lpszOperation)+": contiguous input required"
        );
    }
    if( (c_mInput.size(0)>INT_MAX)||
        (c_mInput.numel()>static_cast<std::size_t>(INT_MAX)) )
    {
        throw std::overflow_error(
            std::string(c_lpszOperation)+": tensor is too large"
        );
    }

    if( _lpmGamma!=nullptr )
    {
        _validateParameter( _lpmGamma,c_mInput.size(0),"gamma" );
        _validateParameter( _lpmBeta,c_mInput.size(0),"beta" );
    }
    return {
        static_cast<int>(c_mInput.size(0)),
        static_cast<int>(
            c_mInput.numel()/static_cast<std::size_t>(c_mInput.size(0))
        )
    };
}

void LayerNorm::backward(
    const TensorGradList& c_lpmOutputGrads,
    const TensorList& c_spmInputs,
    const TensorList& c_spmOutputs
)
{
    (void)c_spmOutputs;
    const ShapeInfo shape =_validateInput(
        c_spmInputs,"LayerNorm::backward"
    );
    const cufMat& c_mOutputGrad =requireSingleOutputGrad(
        c_lpmOutputGrads,"LayerNorm::backward"
    );
    if( c_mOutputGrad.shape()!=c_spmInputs[0]->_mData.shape() )
    {
        throw std::invalid_argument(
            "LayerNorm::backward: gradient shape mismatch"
        );
    }
    if( !c_mOutputGrad.isContiguous() )
    {
        throw std::invalid_argument(
            "LayerNorm::backward: contiguous gradient required"
        );
    }

    cuda_LayerNorm_backward(
        c_spmInputs[0]->_mGrad,
        _lpmGamma==nullptr ? nullptr : &_lpmGamma->_mGrad,
        _lpmBeta==nullptr ? nullptr : &_lpmBeta->_mGrad,
        c_mOutputGrad,
        c_spmInputs[0]->_mData,
        _lpmGamma==nullptr ? nullptr : &_lpmGamma->_mData,
        shape.nFeatures,
        shape.nPositions,
        _fEpsilon
    );
}

TensorList LayerNorm::forward(const TensorList& c_spmInputs)
{
    const ShapeInfo shape =_validateInput(
        c_spmInputs,"LayerNorm::forward"
    );
    auto spmResult =std::make_shared<Tensor>(
        c_spmInputs[0]->_mData.shape()
    );
    cuda_LayerNorm_forward(
        spmResult->_mData,
        c_spmInputs[0]->_mData,
        _lpmGamma==nullptr ? nullptr : &_lpmGamma->_mData,
        _lpmBeta==nullptr ? nullptr : &_lpmBeta->_mData,
        shape.nFeatures,
        shape.nPositions,
        _fEpsilon
    );
    return {spmResult};
}
