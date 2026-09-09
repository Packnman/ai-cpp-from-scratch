#include "cuda_function_Embedding.h"

#include <stdexcept>

#include "cuda_tensor.h"

Embedding::Embedding(Tensor* lpWeight)
    :_lpmWeight( lpWeight )
{
    if( _lpmWeight==nullptr )
    {
        throw std::invalid_argument("Embedding: weight must not be null");
    }
    if( (_lpmWeight->_mData.dim()!=2)||
        (_lpmWeight->_mData.size(0)<=0)||
        (_lpmWeight->_mData.size(1)<=0) )
    {
        throw std::invalid_argument(
            "Embedding: weight shape must be [embeddingSize,vocabSize]"
        );
    }
}

TensorList Embedding::forward(
    const std::shared_ptr<const cunMat>& c_spmIndices
)
{
    if( c_spmIndices==nullptr )
    {
        throw std::invalid_argument("Embedding::forward: indices must not be null");
    }
    if( c_spmIndices->dim()!=2 )
    {
        throw std::invalid_argument(
            "Embedding::forward: index shape must be [sequence,batch]"
        );
    }

    auto output =std::make_shared<Tensor>(
        std::vector<std::int64_t>{
            _lpmWeight->_mData.size(0),
            c_spmIndices->size(0),
            c_spmIndices->size(1)
        }
    );
    cuda_Embedding_forward(
        output->_mData,
        _lpmWeight->_mData,
        *c_spmIndices
    );
    return {output};
}

void Embedding::backward(
    const TensorGradList& c_lpmOutputGrads,
    const std::shared_ptr<const cunMat>& c_spmIndices,
    const TensorList& c_spmOutputs
)
{
    (void)c_spmOutputs;
    //
    if( (c_lpmOutputGrads.size()!=1)||
        (c_lpmOutputGrads[0]==nullptr) )
    {
        throw std::runtime_error(
            "Embedding::backward: exactly one output gradient is required"
        );
    }
    if( c_spmIndices==nullptr )
    {
        throw std::runtime_error(
            "Embedding::backward: saved indices are missing"
        );
    }

    cuda_Embedding_backward(
        _lpmWeight->_mGrad,
        *c_lpmOutputGrads[0],
        *c_spmIndices
    );
}
