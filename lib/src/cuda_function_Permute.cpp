#include "cuda_function_Permute.h"

#include <stdexcept>
#include <utility>
#include <vector>

#include "cuda_tensor.h"


std::vector<std::size_t> Permute::_dimensionsFor(
    const std::vector<std::size_t>& c_nConfigured,
    std::size_t c_nRank
)
{
    if( c_nConfigured.empty() )
    {
        std::vector<std::size_t> nIdentity(c_nRank);
        for( std::size_t nDimension=0;nDimension<c_nRank;++nDimension )
        {
            nIdentity[nDimension] =nDimension;
        }
        return nIdentity;
    }
    if( c_nConfigured.size()!=c_nRank )
    {
        throw std::invalid_argument("Permute: rank mismatch");
    }
    return c_nConfigured;
}

std::vector<std::size_t> Permute::_inverseOf(
    const std::vector<std::size_t>& c_nDimensions
)
{
    std::vector<std::size_t> nInverse(c_nDimensions.size());
    for( std::size_t nOutput=0;nOutput<c_nDimensions.size();++nOutput )
    {
        nInverse[c_nDimensions[nOutput]] =nOutput;
    }
    return nInverse;
}

Permute::Permute()
    :Function()
{
}

Permute::Permute(std::vector<std::size_t> nDimensions)
    :_nDimensions( std::move(nDimensions) )
{
    std::vector<bool> isUsed(_nDimensions.size(),false);
    for( std::size_t nDimension : _nDimensions )
    {
        if( (nDimension>=_nDimensions.size())||isUsed[nDimension] )
        {
            throw std::invalid_argument("Permute: invalid permutation");
        }
        isUsed[nDimension] =true;
    }
}

Permute::~Permute()
{
}

void Permute::backward(
    const TensorGradList& c_lpmOutputGrads,
    const TensorList& c_spmInputs,
    const TensorList& c_spmOutputs
)
{
    if( (c_spmInputs.size()!=1)||(c_spmInputs[0]==nullptr) )
    {
        throw std::runtime_error(
            "Permute::backward: exactly one input is required"
        );
    }
    const cufMat& c_mGrad =requireSingleOutputGrad(
        c_lpmOutputGrads,"Permute::backward"
    );
    const auto nDimensions =_dimensionsFor(
        _nDimensions,c_spmInputs[0]->_mData.dim()
    );
    const auto nInverse =_inverseOf(nDimensions);
    if( (c_spmOutputs.size()!=1)||(c_spmOutputs[0]==nullptr)||
        (c_mGrad.shape()!=c_spmOutputs[0]->_mData.shape()) )
    {
        throw std::invalid_argument(
            "Permute::backward: gradient shape mismatch"
        );
    }

    cufMat mRestored(c_spmInputs[0]->_mData.shape());
    cuda_Permute(
        mRestored,
        c_mGrad,nInverse
    );
    cuda_axpy(
        c_spmInputs[0]->_mGrad,
        1.0f,
        mRestored
    );
}

TensorList Permute::forward(const TensorList& c_spmInputs)
{
    if( (c_spmInputs.size()!=1)||(c_spmInputs[0]==nullptr) )
    {
        throw std::runtime_error(
            "Permute::forward: exactly one input is required"
        );
    }
    const auto nDimensions =_dimensionsFor(
        _nDimensions,c_spmInputs[0]->_mData.dim()
    );
    std::vector<std::int64_t> shapeOutput(nDimensions.size());
    for( std::size_t nOutput=0;nOutput<nDimensions.size();++nOutput )
    {
        shapeOutput[nOutput] =
            c_spmInputs[0]->_mData.size(nDimensions[nOutput]);
    }

    auto spmResult =std::make_shared<Tensor>(shapeOutput);
    cuda_Permute(
        spmResult->_mData,
        c_spmInputs[0]->_mData,nDimensions
    );
    //
    return {spmResult};
}
