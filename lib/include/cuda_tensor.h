#pragma once
#include <vector>
#include <unordered_set>
#include "cuda_matrix.h"
#include "cuda_function.h"

class Tensor{
public:
    Tensor(int nRows,int nCols);
    explicit Tensor(const std::vector<std::int64_t>& shape);
    Tensor(std::initializer_list<std::int64_t> shape);
    ~Tensor();
private:
public:
    cufMat   _mData; // 行列内容
    cufMat   _mGrad; // 微分値

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
