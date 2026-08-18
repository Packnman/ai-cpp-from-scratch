#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

class cuCUBLAS{
public:
    cuCUBLAS()  {cublasCreate( &_hCuda );}
    ~cuCUBLAS() {cublasDestroy( _hCuda );}

    cublasHandle_t get() const  {return _hCuda;}
private:
    cublasHandle_t _hCuda;
};

cublasHandle_t getCublasHandle()
{
    static cuCUBLAS cuCublas;
    return cuCublas.get();
}
