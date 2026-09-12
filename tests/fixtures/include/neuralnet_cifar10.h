#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "cuda_function_Conv2D.h"
#include "cuda_function_ReLU.h"
#include "cuda_function_Pooling.h"
#include "cuda_function_Linear.h"
#include "cuda_function_BatchNorm.h"
#include "cuda_function_Dropout.h"
#include "cuda_function_SoftmaxCrossEntropy.h"
#include "cuda_tensor.h"
#include "module.h"


class LayerConv2D : public Module
{
public:
    LayerConv2D(
        int nInputChannels,
        int nOutputChannels,
        int nInputHeight,
        int nInputWidth,
        int nKernelSize,
        int nStride =1,
        int nPadding =0
    );
    ~LayerConv2D() override;

private:
    std::shared_ptr<Tensor> _spmWeight;
    std::shared_ptr<Tensor> _spmBias;
    Conv2D _cnvConv2D;
    int _nOutputChannels;
    int _nOutputHeight;
    int _nOutputWidth;
    int _nFanIn;
public: // propaties
    int outputChannels() const;
    int outputHeight() const;
    int outputWidth() const;

private:
    int _checkedWeightRows(int nOutputChannels,int nInputChannels,int nKernelSize) const;
    int _checkedOutputChannels(int nOutputChannels) const;
    int _outputSize(int nInputSize,int nKernelSize,int nStride,int nPadding) const;
    int _fanIn(int nInputChannels,int nKernelSize) const;
public:
    void init(std::mt19937& rngRandom);
    //
    std::shared_ptr<Tensor> forward(
        std::vector<std::shared_ptr<Tensor>>& spmInputs
    ) override;
};

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
    Pooling _mplPool;
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

class NeuralNet_Cifar10 : public Model
{
public:
    explicit NeuralNet_Cifar10(
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
