#include "cuda_function_Mask.h"

#include <climits>
#include <stdexcept>
#include <string>

#include "cuda_tensor.h"


Mask::Mask(Tensor* lpMask)
    :_lpmMask( lpMask )
{
    if( _lpmMask==nullptr )
    {
        throw std::invalid_argument("Mask: mask must not be null");
    }
}

Mask::~Mask()
{
}

Mask::ShapeInfo Mask::validateInput(
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
    if( _lpmMask==nullptr )
    {
        throw std::invalid_argument(
            std::string(c_lpszOperation)+": mask must not be null"
        );
    }

    const cufMat& c_mInput =c_spmInputs[0]->_mData;
    const cufMat& c_mMask =_lpmMask->_mData;
    if( (c_mInput.dim()<2)||(c_mInput.size(0)<=0)||
        (c_mInput.size(1)<=0) )
    {
        throw std::invalid_argument(
            std::string(c_lpszOperation)+": rank-two input is required"
        );
    }

    bool isBroadcast =false;
    if( c_mMask.shape()!=c_mInput.shape() )
    {
        isBroadcast =
            (c_mMask.dim()==2)&&
            (c_mMask.size(0)==c_mInput.size(0))&&
            (c_mMask.size(1)==c_mInput.size(1));
        if( !isBroadcast )
        {
            throw std::invalid_argument(
                std::string(c_lpszOperation)+": mask shape mismatch"
            );
        }
    }
    if( !c_mInput.isContiguous()||!c_mMask.isContiguous()||
        !c_spmInputs[0]->_mGrad.isContiguous() )
    {
        throw std::invalid_argument(
            std::string(c_lpszOperation)+": contiguous tensors required"
        );
    }
    if( (c_mInput.size(0)>INT_MAX)||(c_mInput.size(1)>INT_MAX)||
        (c_mInput.numel()>static_cast<std::size_t>(INT_MAX))||
        (c_mMask.numel()>static_cast<std::size_t>(INT_MAX)) )
    {
        throw std::overflow_error(
            std::string(c_lpszOperation)+": tensor is too large"
        );
    }

    const std::size_t nQuery =
        static_cast<std::size_t>(c_mInput.size(0));
    const std::size_t nKey =
        static_cast<std::size_t>(c_mInput.size(1));
    return {
        static_cast<int>(nQuery),
        static_cast<int>(nKey),
        static_cast<int>(c_mInput.numel()/(nQuery*nKey)),
        isBroadcast
    };
}

void Mask::backward(
    const TensorGradList& c_lpmOutputGrads,
    const TensorList& c_spmInputs,
    const TensorList& c_spmOutputs
)
{
    (void)c_spmOutputs;
    const ShapeInfo shape =validateInput(
        c_spmInputs,"Mask::backward"
    );
    const cufMat& c_mGrad =requireSingleOutputGrad(
        c_lpmOutputGrads,"Mask::backward"
    );
    if( c_mGrad.shape()!=c_spmInputs[0]->_mData.shape() )
    {
        throw std::invalid_argument("Mask::backward: gradient shape mismatch");
    }
    if( !c_mGrad.isContiguous() )
    {
        throw std::invalid_argument(
            "Mask::backward: contiguous gradient required"
        );
    }

    cuda_Mask_backward(
        c_spmInputs[0]->_mGrad,
        c_mGrad,
        _lpmMask->_mData,
        shape._nQuery,
        shape._nKey,
        shape._nTrailing,
        shape._isBroadcast
    );
}

TensorList Mask::forward(const TensorList& c_spmInputs)
{
    const ShapeInfo shape =validateInput(
        c_spmInputs,"Mask::forward"
    );
    auto spmResult =std::make_shared<Tensor>(
        c_spmInputs[0]->_mData.shape()
    );
    cuda_Mask_forward(
        spmResult->_mData,
        c_spmInputs[0]->_mData,
        _lpmMask->_mData,
        shape._nQuery,
        shape._nKey,
        shape._nTrailing,
        shape._isBroadcast
    );
    return {spmResult};
}
