#include "cuda_function_Reshape.h"

#include <limits>
#include <stdexcept>
#include <vector>

#include "cuda_tensor.h"

std::vector<std::int64_t> Reshape::resolveShape(
    const std::vector<std::int64_t>& c_nConfigured,
    const cufMat& c_mInput
)
{
    if( c_nConfigured.empty() )
    {
        return c_mInput.shape();
    }

    std::vector<std::int64_t> nResult =c_nConfigured;
    std::size_t nKnown =1;
    std::size_t nInferred =nResult.size();
    for( std::size_t nDimension=0;nDimension<nResult.size();++nDimension )
    {
        const std::int64_t nExtent =nResult[nDimension];
        if( nExtent==-1 )
        {
            if( nInferred!=nResult.size() )
            {
                throw std::invalid_argument("Reshape: multiple inferred dimensions");
            }
            nInferred =nDimension;
            continue;
        }
        if( nExtent<0 )
        {
            throw std::invalid_argument("Reshape: negative shape extent");
        }
        if( (nExtent!=0)&&
            (static_cast<std::size_t>(nExtent)>
             std::numeric_limits<std::size_t>::max()/nKnown) )
        {
            throw std::overflow_error("Reshape: shape is too large");
        }
        nKnown *=static_cast<std::size_t>(nExtent);
    }

    const std::size_t nElements =c_mInput.numel();
    if( nInferred!=nResult.size() )
    {
        if( (nKnown==0)||(nElements%nKnown!=0)||
            (nElements/nKnown>
             static_cast<std::size_t>(
                 std::numeric_limits<std::int64_t>::max()
             )) )
        {
            throw std::invalid_argument("Reshape: cannot infer target shape");
        }
        nResult[nInferred] =static_cast<std::int64_t>(nElements/nKnown);
    }

    std::size_t nTargetElements =1;
    for( std::int64_t nExtent : nResult )
    {
        if( nExtent==0 )
        {
            nTargetElements =0;
            break;
        }
        if( static_cast<std::size_t>(nExtent)>
            std::numeric_limits<std::size_t>::max()/nTargetElements )
        {
            throw std::overflow_error("Reshape: shape is too large");
        }
        nTargetElements *=static_cast<std::size_t>(nExtent);
    }
    if( nTargetElements!=nElements )
    {
        throw std::invalid_argument("Reshape: element count mismatch");
    }
    return nResult;
}

Reshape::Reshape()
    :Function()
{
}

Reshape::Reshape(std::vector<std::int64_t> nShape)
    :_nShape( std::move(nShape) )
{
}

Reshape::~Reshape()
{
}

void Reshape::backward(
    const TensorGradList& c_lpmOutputGrads,
    const TensorList& c_spmInputs,
    const TensorList& c_spmOutputs
)
{
    if( (c_spmInputs.size()!=1)||(c_spmInputs[0]==nullptr) )
    {
        throw std::runtime_error("Reshape::backward: exactly one input is required");
    }
    const cufMat& c_mGrad =requireSingleOutputGrad(
        c_lpmOutputGrads,"Reshape::backward"
    );
    if( (c_spmOutputs.size()!=1)||(c_spmOutputs[0]==nullptr)||
        (c_mGrad.shape()!=c_spmOutputs[0]->_mData.shape()) )
    {
        throw std::invalid_argument("Reshape::backward: gradient shape mismatch");
    }
    const cufMat mRestored =c_mGrad.reshape(
        c_spmInputs[0]->_mData.shape()
    );
    cuda_axpy( c_spmInputs[0]->_mGrad,1.0f,mRestored );
}

TensorList Reshape::forward(const TensorList& c_spmInputs)
{
    if( (c_spmInputs.size()!=1)||(c_spmInputs[0]==nullptr) )
    {
        throw std::runtime_error("Reshape::forward: exactly one input is required");
    }
    if( !c_spmInputs[0]->_mData.isContiguous() )
    {
        throw std::invalid_argument("Reshape::forward: contiguous input required");
    }
    const auto nShape =resolveShape(_nShape,c_spmInputs[0]->_mData);
    const cufMat mReshaped(c_spmInputs[0]->_mData.reshape(nShape));
    auto spmResult =std::make_shared<Tensor>( nShape );
    spmResult->_mData =mReshaped;
    return {spmResult};
}
