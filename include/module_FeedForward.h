#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "cuda_function.h"
#include "module.h"


class FeedForward : public Module
{
public:
    FeedForward(
        int nEmbeddingSize,
        int nHiddenSize,
        float fDropoutProbability =0.0f,
        std::uint64_t nDropoutSeed =0
    );
    ~FeedForward() override;

private:
    int _nEmbeddingSize;
    int _nHiddenSize;
    float _fDropoutProbability;

    std::shared_ptr<Tensor> _spmWeight1;
    std::shared_ptr<Tensor> _spmBias1;
    std::shared_ptr<Tensor> _spmWeight2;
    std::shared_ptr<Tensor> _spmBias2;

    Linear _lnrInput;
    GELU _gelActivation;
    Linear _lnrOutput;
    Dropout _drpOutput;

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
