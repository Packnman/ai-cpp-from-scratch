#pragma once
#include "cuMat.h"
class Function;

class Tensor{
public:
    Tensor(Function* func,int rows,int cols);
    ~Tensor();
private:
public:
    cuMat       _mData;
    cuMat       _mGrad;
    Function*   _lpFunc;
public:
};
