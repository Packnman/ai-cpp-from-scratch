#pragma once
#include <random>
#include <memory>
#include "cuda_tensor.h"
#include "cuda_function.h"
#include "module.h"

// --------------------------
// MnistInputLayer
// --------------------------
// 役割：
//  ・前処理
// --------------------------
class MnistInputLayer : public Module{
public:
    MnistInputLayer();
    ~MnistInputLayer() override;

public:
    void init(std::mt19937& rngRandom);
    //
    std::shared_ptr<Tensor> forward(
        std::vector<std::shared_ptr<Tensor>>& spmInputs
    ) override;
};


// --------------------------
// MnistHiddenLayer
// --------------------------
// 役割：
//  ・1層目からn-1層目までの処理
// --------------------------
class MnistHiddenLayer : public Module{
public:
    MnistHiddenLayer(int nInput,int nOutput,float fDropoutRate,std::uint64_t nDropoutSeed);
    ~MnistHiddenLayer() override;

private:
    std::shared_ptr<Tensor> _spmWeight;
    std::shared_ptr<Tensor> _spmBias;
    // BatchNormのためのメンバー変数
    std::shared_ptr<Tensor> _spmGamma;
    std::shared_ptr<Tensor> _spmBeta;
    std::shared_ptr<Tensor> _spmRunningMean;
    std::shared_ptr<Tensor> _spmRunningVar;
    //
    Linear      _lnrLinear;
    BatchNorm   _bnmBatchNorm;
    ReLU        _rluReLU;
    Dropout     _drpDropout;

public:
    void init(std::mt19937& rngRandom);
    //
    std::shared_ptr<Tensor> forward(
        std::vector<std::shared_ptr<Tensor>>& spmInputs
    ) override;
};

// --------------------------
// MnistOutputLayer
// --------------------------
// 役割：
//  ・n層目の処理
// --------------------------
class MnistOutputLayer : public Module{
public:
    MnistOutputLayer(int nInput,int nOutput);
    ~MnistOutputLayer() override;

private:
    std::shared_ptr<Tensor> _spmWeight;
    std::shared_ptr<Tensor> _spmBias;
    //
    Linear  _lnrLinear;

public:
    void init(std::mt19937& rngRandom);
    //
    std::shared_ptr<Tensor> forward(
        std::vector<std::shared_ptr<Tensor>>& spmInputs
    ) override;
};

// --------------------------
// MnistNeuralNet
// --------------------------
// 役割：
//  ・
// --------------------------
class MnistNeuralNet : public Model{
public:
    MnistNeuralNet(std::uint32_t nSeed,float fDropoutRate=0.2f);
    ~MnistNeuralNet() override;
private:
    MnistInputLayer  _lyrInput;
    MnistHiddenLayer _lyrHidden1;
    MnistHiddenLayer _lyrHidden2;
    MnistOutputLayer _lyrOutput;
    //
    SoftmaxCrossEntropy     _sceEntropy;
public:
    std::shared_ptr<Tensor> forward(
        std::vector<std::shared_ptr<Tensor>>& spmInputs
    ) override;
    //
    std::shared_ptr<Tensor> loss(
        const std::shared_ptr<Tensor>& c_spmInput,
        const std::shared_ptr<Tensor>& c_spmTarget
    );
};
