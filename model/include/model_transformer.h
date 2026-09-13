#pragma once

#include "module_TransformerBlock.h"

struct TransformerConfig
{
    int nVocabulary = 0;
    int nBlocks = 4;
    int nEmbedding = 256;
    int nHeads = 4;
    int nHidden = 1024;
    int nContext = 128;
    float fDropout = 0.1f;
    std::uint64_t nSeed = 42;
    void validate() const;
};

class Transformer final : public Model
{
public:
    explicit Transformer( const TransformerConfig& c_cfgConfig );
    ~Transformer() override = default;
    const TransformerConfig& config() const;
    std::vector<std::uint64_t> dropoutCounters() const;
    void setDropoutCounters( const std::vector<std::uint64_t>& c_nCounters );
    TensorPtr forward( TensorList& spmInputs ) override;
    TensorPtr forward( const std::shared_ptr<const cunMat>& c_spmIds );
    TensorPtr loss(
        const std::shared_ptr<const cunMat>& c_spmIds,
        const cunMat& c_mTargets,
        int nPadId = 0
    );

private:
    TransformerConfig _cfgConfig;
    Tensor _tenTokens;
    Tensor _tenPositions;
    Tensor _tenGamma;
    Tensor _tenBeta;
    Tensor _tenOutputWeight;
    Tensor _tenOutputBias;
    Embedding _embTokens;
    Embedding _embPositions;
    LayerNorm _lynFinal;
    Linear _lnrOutput;
    Add _addPositions;
    std::vector<std::unique_ptr<TransformerBlock>> _spBlocks;
};
