#include <iostream>
#include <stdio.h>
#include "tensor.h"


Tensor::Tensor(Function* func,int rows,int cols)
{
    _lpFunc =func;
    _mData  =cuMat( rows,cols );
    _mGrad  =cuMat( rows,cols );
}
Tensor::~Tensor()
{
    
}