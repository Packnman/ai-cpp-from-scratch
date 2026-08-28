#pragma once

#include <memory>
#include "cuda_tensor.h"
#include "cuda_function.h"
#include "graph.h"

// --------------------------
// LayerInput
// --------------------------
// 役割：
//  ・前処理
// --------------------------
class LayerInput : public Graph{
public:
    LayerInput();
    ~LayerInput();

public:
    void init(std::mt19937& randam);
    //
    std::shared_ptr<Tensor> forward(
        std::vector<std::shared_ptr<Tensor>>& inputs
    );
    std::vector<Tensor*> getParams();
};


// --------------------------
// LayerHidden
// --------------------------
// 役割：
//  ・1層目からn-1層目までの処理
// --------------------------
class LayerHidden : public Graph{
public:
    LayerHidden(int nInput,int nOutput);
    ~LayerHidden();

private:
    std::shared_ptr<Tensor> _spmWeight;
    std::shared_ptr<Tensor> _spmBias;
    //
    Linear  _linear;
    ReLU    _relu;

public:
    void init(std::mt19937& randam);
    //
    std::shared_ptr<Tensor> forward(
        std::vector<std::shared_ptr<Tensor>>& inputs
    );
    std::vector<Tensor*> getParams();
};

// --------------------------
// LayerOutput
// --------------------------
// 役割：
//  ・n層目の処理
// --------------------------
class LayerOutput : public Graph{
public:
    LayerOutput(int nInput,int nOutput);
    ~LayerOutput();

private:
    std::shared_ptr<Tensor> _spmWeight;
    std::shared_ptr<Tensor> _spmBias;
    //
    Linear  _linear;

public:
    void init(std::mt19937& randam);
    //
    std::shared_ptr<Tensor> forward(
        std::vector<std::shared_ptr<Tensor>>& inputs
    );
    std::vector<Tensor*> getParams();
};

// --------------------------
// NeuralNet
// --------------------------
// 役割：
//  ・
// --------------------------
class NeuralNet : public Model{
public:
    NeuralNet(std::uint32_t seed);
    ~NeuralNet();
private:
    LayerInput  _lyrInput;
    LayerHidden _lyrHidden1;
    LayerHidden _lyrHidden2;
    LayerOutput _lyrOutput;
    //
    SoftmaxCrossEntropy     _entropy;
public:
    void load(const char* szFName);
    void save(const char* szFName);
    //
    std::shared_ptr<Tensor> forward(
        std::vector<std::shared_ptr<Tensor>>& inputs
    );
    std::shared_ptr<Tensor> loss(
        const std::shared_ptr<Tensor>& input,
        const std::shared_ptr<Tensor>& target
    );
    std::vector<Tensor*> getParams();
};