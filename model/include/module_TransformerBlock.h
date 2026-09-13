#pragma once

#include "module_Attention.h"
#include "module_FeedForward.h"

class TransformerBlock final : public Module
{
public:
    TransformerBlock(
        int nEmbedding,
        int nHeads,
        int nHidden,
        float fDropout = 0.1f,
        std::uint64_t nSeed = 42
    );

private:
    Tensor _tenGamma1;
    Tensor _tenBeta1;
    Tensor _tenGamma2;
    Tensor _tenBeta2;
    LayerNorm _lynAttention;
    LayerNorm _lynFeedForward;
    Attention _attAttention;
    FeedForward _ffdFeedForward;
    Add _addResidual;
    
public:
    void init( std::mt19937& rngRandom );
    std::vector<std::uint64_t> dropoutCounters() const;
    void setDropoutCounters( const std::vector<std::uint64_t>& c_nCounters );
    TensorPtr forward( TensorList& spmInputs ) override;
};
