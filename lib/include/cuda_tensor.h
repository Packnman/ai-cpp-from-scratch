#pragma once
#include <vector>
#include <unordered_set>
#include "cuda_matrix.h"
#include "cuda_function.h"

class Tensor{
public:
    Tensor(int nRows,int nCols);
    ~Tensor();
private:
public:
    cuMat   _mData; // 行列内容
    cuMat   _mGrad; // 微分値

    std::shared_ptr<Context> _spContext;   // 使用関数
public:
    void backward();
private:
    void buildBackwardGraph(
        Tensor* lpValue,
        std::vector<Context*>& lpContexts,
        std::unordered_set<Context*>& lpVisited
    );
};
