#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

class cuCublas{
public:
    cuCublas()  {cublasCreate( &_cblHandle );}
    ~cuCublas() {cublasDestroy( _cblHandle );}

    cublasHandle_t get() const  {return _cblHandle;}
private:
    cublasHandle_t _cblHandle;
};

cublasHandle_t getCublasHandle()
{
    static cuCublas cblInstance;
    return cblInstance.get();
}
