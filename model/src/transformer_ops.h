#pragma once

#include "cuda_tensor.h"
#include <cmath>
#include <random>
#include <utility>

// Context owns graph-local operations; model parameters must outlive the graph.
template <class Operation, class... Arguments>
TensorPtr g_apply( const TensorList& c_spmInputs, Arguments&&... argArguments )
{
    auto spFunction = std::make_shared<Operation>( std::forward<Arguments>( argArguments )... );
    auto spmResult = ( *spFunction )( c_spmInputs );
    spmResult->_spContext->_spFunction = std::move( spFunction );
    return spmResult;
}

inline TensorPtr g_reshape( const TensorPtr& c_spmInput, std::vector<std::int64_t> nShape )
{
    return g_apply<Reshape>( { c_spmInput }, std::move( nShape ) );
}

inline TensorPtr g_permute( const TensorPtr& c_spmInput, std::vector<std::size_t> nAxes )
{
    return g_apply<Permute>( { c_spmInput }, std::move( nAxes ) );
}

inline void g_initialize( Tensor& tenWeight, int nFanIn, std::mt19937& rngRandom )
{
    std::normal_distribution<float> dstNormal( 0.0f,
                                               1.0f / std::sqrt( static_cast<float>( nFanIn ) ) );
    std::vector<float> fValues( tenWeight._mData.numel() );
    for( float& fValue : fValues )
    {
        fValue = dstNormal( rngRandom );
    }
    tenWeight._mData.copyFromHost( fValues.data(), fValues.size() );
}

class CausalMask final : public Mask
{
    TensorPtr _spmMask;

public:
    explicit CausalMask( const TensorPtr& c_spmMask )
        : Mask( c_spmMask.get() ), _spmMask( c_spmMask )
    {
    }
};
