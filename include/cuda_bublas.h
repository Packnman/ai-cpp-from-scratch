#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

class cuCublas{
public:
    cuCublas()  {cublasCreate( &_hCuda );}
    ~cuCublas() {cublasDestroy( _hCuda );}

    cublasHandle_t get() const  {return _hCuda;}
private:
    cublasHandle_t _hCuda;
};

cublasHandle_t getCublasHandle()
{
    static cuCublas cuCublas;
    return cuCublas.get();
}
