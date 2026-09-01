#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "cuda_function.h"
#include "cuda_tensor.h"
#include "module.h"

class Cifar10ConvBlock : public Module
{
public:
    Cifar10ConvBlock(
        int nInputChannels,
        int nOutputChannels,
        int nInputHeight,
        int nInputWidth
    );

    void init(std::mt19937& rngRandom);
    int outputChannels() const;
    int outputHeight() const;
    int outputWidth() const;

    std::shared_ptr<Tensor> forward(
        std::vector<std::shared_ptr<Tensor>>& spmInputs
    ) override;

private:
    LayerConv2D _lyrConv;
    ReLU _rluReLU;
    MaxPool2D _mplPool;
    int _nOutputChannels;
    int _nOutputHeight;
    int _nOutputWidth;
};

class Cifar10DenseLayer : public Module
{
public:
    Cifar10DenseLayer(
        int nInput,
        int nOutput,
        float fDropoutRate,
        std::uint64_t nDropoutSeed
    );

    void init(std::mt19937& rngRandom);
    std::shared_ptr<Tensor> forward(
        std::vector<std::shared_ptr<Tensor>>& spmInputs
    ) override;

private:
    std::shared_ptr<Tensor> _spmWeight;
    std::shared_ptr<Tensor> _spmBias;
    std::shared_ptr<Tensor> _spmGamma;
    std::shared_ptr<Tensor> _spmBeta;
    std::shared_ptr<Tensor> _spmRunningMean;
    std::shared_ptr<Tensor> _spmRunningVar;
    Linear _lnrLinear;
    BatchNorm _bnmBatchNorm;
    ReLU _rluReLU;
    Dropout _drpDropout;
};

class Cifar10OutputLayer : public Module
{
public:
    Cifar10OutputLayer(int nInput,int nOutput);

    void init(std::mt19937& rngRandom);
    std::shared_ptr<Tensor> forward(
        std::vector<std::shared_ptr<Tensor>>& spmInputs
    ) override;

private:
    std::shared_ptr<Tensor> _spmWeight;
    std::shared_ptr<Tensor> _spmBias;
    Linear _lnrLinear;
};

class Cifar10NeuralNet : public Model
{
public:
    explicit Cifar10NeuralNet(
        std::uint32_t nSeed,
        float fDropoutRate=0.2f
    );

    std::shared_ptr<Tensor> forward(
        std::vector<std::shared_ptr<Tensor>>& spmInputs
    ) override;

    std::shared_ptr<Tensor> loss(
        const std::shared_ptr<Tensor>& c_spmInput,
        const std::shared_ptr<Tensor>& c_spmTarget
    );

private:
    Cifar10ConvBlock _lyrConv1;
    Cifar10ConvBlock _lyrConv2;
    Cifar10DenseLayer _lyrHidden;
    Cifar10OutputLayer _lyrOutput;
    SoftmaxCrossEntropy _sceEntropy;
};
