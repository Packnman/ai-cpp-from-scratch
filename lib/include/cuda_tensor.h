#pragma once
#include <vector>
#include <unordered_set>
#include "cuda_matrix.h"
#include "cuda_function.h"

class Tensor{
public:
    Tensor(int rows,int cols);
    ~Tensor();
private:
public:
    cuMat   _mData; // 行列内容
    cuMat   _mGrad; // 微分値

    std::shared_ptr<Context> _spContexts;   // 使用関数
public:
    void backward();
private:
    void buildBackwardGraph(
        Tensor* value,
        std::vector<Context*>& contexts,
        std::unordered_set<Context*>& visited
    );
};
