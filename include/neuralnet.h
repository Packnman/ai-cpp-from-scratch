#pragma once
#include <random>
#include <memory>
#include "cuda_tensor.h"
#include "cuda_function.h"
#include "module.h"

// --------------------------
// LayerInput
// --------------------------
// 役割：
//  ・前処理
// --------------------------
class LayerInput : public Module{
public:
    LayerInput();
    ~LayerInput() override;

public:
    void init(std::mt19937& rngRandom);
    //
    std::shared_ptr<Tensor> forward(
        std::vector<std::shared_ptr<Tensor>>& spmInputs
    ) override;
};


// --------------------------
// LayerHidden
// --------------------------
// 役割：
//  ・1層目からn-1層目までの処理
// --------------------------
class LayerHidden : public Module{
public:
    LayerHidden(int nInput,int nOutput);
    ~LayerHidden() override;

private:
    std::shared_ptr<Tensor> _spmWeight;
    std::shared_ptr<Tensor> _spmBias;
    //
    Linear  _lnrLinear;
    ReLU    _rluReLU;

public:
    void init(std::mt19937& rngRandom);
    //
    std::shared_ptr<Tensor> forward(
        std::vector<std::shared_ptr<Tensor>>& spmInputs
    ) override;
};

// --------------------------
// LayerOutput
// --------------------------
// 役割：
//  ・n層目の処理
// --------------------------
class LayerOutput : public Module{
public:
    LayerOutput(int nInput,int nOutput);
    ~LayerOutput() override;

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
// NeuralNet
// --------------------------
// 役割：
//  ・
// --------------------------
class NeuralNet : public Model{
public:
    NeuralNet(std::uint32_t nSeed);
    ~NeuralNet() override;
private:
    LayerInput  _lyrInput;
    LayerHidden _lyrHidden1;
    LayerHidden _lyrHidden2;
    LayerOutput _lyrOutput;
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
