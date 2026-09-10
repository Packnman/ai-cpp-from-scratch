#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "cuda_function.h"
#include "module.h"


class Attention : public Module
{
public:
    Attention(
        int nEmbeddingSize,
        int nHeads,
        float fDropoutProbability =0.0f,
        std::uint64_t nDropoutSeed =0
    );
    ~Attention() override;

private:
    int _nEmbeddingSize;
    int _nHeads;
    int _nHeadSize;
    float _fDropoutProbability;

    std::shared_ptr<Tensor> _spmQueryWeight;
    std::shared_ptr<Tensor> _spmQueryBias;
    std::shared_ptr<Tensor> _spmKeyWeight;
    std::shared_ptr<Tensor> _spmKeyBias;
    std::shared_ptr<Tensor> _spmValueWeight;
    std::shared_ptr<Tensor> _spmValueBias;
    std::shared_ptr<Tensor> _spmOutputWeight;
    std::shared_ptr<Tensor> _spmOutputBias;

    Linear _lnrQuery;
    Linear _lnrKey;
    Linear _lnrValue;
    Linear _lnrOutput;
    BatchMatMul _bmmQueryKey;
    Scale _sclScores;
    Softmax _sftWeights;
    BatchMatMul _bmmAttentionValue;
    Dropout _drpAttention;

    static void _initializeWeight(
        Tensor& mWeight,
        int nFanIn,
        std::mt19937& rngRandom
    );

public:
    void init(std::mt19937& rngRandom);
    std::shared_ptr<Tensor> forward(
        std::vector<std::shared_ptr<Tensor>>& spmInputs
    ) override;
};
