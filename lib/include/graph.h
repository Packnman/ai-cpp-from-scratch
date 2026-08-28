#pragma once

#include <vector>
#include <memory>
#include "cuda_function.h"
#include "cuda_tensor.h"

// --------------------------
// Graph
// --------------------------
// 役割：
//  ・複数のFunctionをまとめて1つの機能として扱う
//  ・学習パラメータを管理する
//  ・forwardの接続関係を定義する
//  ・勾配の初期化などをまとめて行う
// --------------------------
class Graph{
public:
    Graph();
    virtual ~Graph();
public:

public:
    virtual void zero_grads();      // モデル全体の勾配を0にする
    virtual void reset_state();     // モデル全体の内部状態を初期化する
    // 順伝播
    virtual std::shared_ptr<Tensor> forward(
        std::vector<std::shared_ptr<Tensor>>& inputs
    );
    // モデル全体の学習パラメータを返す
    virtual std::vector<Tensor*> getParams();
};




// --------------------------
// Model
// --------------------------
// 役割：
//  ・複数のGraphをまとめてモデル全体を構成する
//  ・モデル全体の学習パラメータを取得する
//  ・モデル全体の勾配を初期化する
//  ・モデル全体の内部状態を初期化する
// --------------------------
class Model
{
public:
    Model();
    virtual ~Model();

public:
    virtual void save(const char* szFName);
    virtual void load(const char* szFName);
    virtual void zero_grads();
    virtual void reset_state();
    // 順伝播
    virtual std::shared_ptr<Tensor> forward(
        std::vector<std::shared_ptr<Tensor>>& inputs
    );
    // モデル全体の学習パラメータを返す
    virtual std::vector<Tensor*> getParams();

};