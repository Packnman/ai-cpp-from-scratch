#include "cuda_function_Softmax.h"

#include <climits>
#include <stdexcept>
#include <string>

#include "cuda_tensor.h"


Softmax::Softmax(std::size_t nAxis)
    :_nAxis( nAxis )
{
}

Softmax::~Softmax()
{
}

Softmax::ShapeInfo Softmax::validateInput(
    const cufMat& c_mInput,
    const char* c_lpszOperation
) const
{
    const auto& c_nShape =c_mInput.shape();
    if( c_nShape.empty()||(_nAxis>=c_nShape.size())||
        (c_nShape[_nAxis]<=0) )
    {
        throw std::invalid_argument(
            std::string(c_lpszOperation)+": invalid axis"
        );
    }
    if( !c_mInput.isContiguous() )
    {
        throw std::invalid_argument(
            std::string(c_lpszOperation)+": contiguous tensor required"
        );
    }
    if( c_mInput.numel()>static_cast<std::size_t>(INT_MAX) )
    {
        throw std::overflow_error(
            std::string(c_lpszOperation)+": tensor is too large"
        );
    }

    std::size_t nOuter =1;
    std::size_t nInner =1;
    for( std::size_t nDimension=0;nDimension<_nAxis;++nDimension )
    {
        nOuter *=static_cast<std::size_t>(c_nShape[nDimension]);
    }
    for( std::size_t nDimension=_nAxis+1;
         nDimension<c_nShape.size();
         ++nDimension )
    {
        nInner *=static_cast<std::size_t>(c_nShape[nDimension]);
    }
    const std::size_t nAxisSize =
        static_cast<std::size_t>(c_nShape[_nAxis]);
    if( (nOuter>static_cast<std::size_t>(INT_MAX))||
        (nInner>static_cast<std::size_t>(INT_MAX))||
        (nAxisSize>static_cast<std::size_t>(INT_MAX))||
        (nOuter*nInner>static_cast<std::size_t>(INT_MAX)) )
    {
        throw std::overflow_error(
            std::string(c_lpszOperation)+": tensor is too large"
        );
    }
    return {
        static_cast<int>(nOuter),
        static_cast<int>(nAxisSize),
        static_cast<int>(nInner),
        static_cast<int>(nOuter*nInner)
    };
}

void Softmax::backward(
    const TensorGradList& c_lpmOutputGrads,
    const TensorList& c_spmInputs,
    const TensorList& c_spmOutputs
)
{
    if( (c_spmInputs.size()!=1)||(c_spmInputs[0]==nullptr) )
    {
        throw std::runtime_error(
            "Softmax::backward: exactly one input is required"
        );
    }
    const ShapeInfo shape =validateInput(
        c_spmInputs[0]->_mData,"Softmax::backward"
    );
    const cufMat& c_mGrad =requireSingleOutputGrad(
        c_lpmOutputGrads,"Softmax::backward"
    );
    if( (c_spmOutputs.size()!=1)||(c_spmOutputs[0]==nullptr)||
        (c_mGrad.shape()!=c_spmInputs[0]->_mData.shape())||
        (c_spmOutputs[0]->_mData.shape()!=c_mGrad.shape()) )
    {
        throw std::invalid_argument(
            "Softmax::backward: gradient shape mismatch"
        );
    }
    if( !c_mGrad.isContiguous()||
        !c_spmOutputs[0]->_mData.isContiguous()||
        !c_spmInputs[0]->_mGrad.isContiguous() )
    {
        throw std::invalid_argument(
            "Softmax::backward: contiguous tensors required"
        );
    }

    cuda_Softmax_backward(
        c_spmInputs[0]->_mGrad,c_mGrad,c_spmOutputs[0]->_mData,
        shape._nOuter,shape._nAxisSize,shape._nInner,shape._nSlices
    );
}

TensorList Softmax::forward(const TensorList& c_spmInputs)
{
    if( (c_spmInputs.size()!=1)||(c_spmInputs[0]==nullptr) )
    {
        throw std::runtime_error(
            "Softmax::forward: exactly one input is required"
        );
    }
    const ShapeInfo shape =validateInput(
        c_spmInputs[0]->_mData,"Softmax::forward"
    );
    auto spmResult =std::make_shared<Tensor>(
        c_spmInputs[0]->_mData.shape()
    );
    cuda_Softmax_forward(
        spmResult->_mData,c_spmInputs[0]->_mData,
        shape._nOuter,shape._nAxisSize,shape._nInner,shape._nSlices
    );
    return {spmResult};
}
