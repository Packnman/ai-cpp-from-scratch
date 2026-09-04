#pragma once

#include <cstdint>
#include <curand.h>
#include <vector>
#include <memory>
#include "cuda_matrix.h"

class Tensor;

// --------------------------
// Function
// --------------------------
class Function{
public:
    Function();
    virtual ~Function();

protected:
    void checkCurand(curandStatus_t crnStatus,const char* c_lpszOperation);
    const cuMat& requireSingleOutputGrad(
        const std::vector<const cuMat*>& c_lpmOutputGrads,
        const char* c_lpszFunctionName
    );
    const cuMat& singleGrad(
        const std::vector<const cuMat*>& c_lpmOutputGrads,
        const char* c_lpszFunctionName
    );

public:
    virtual void backward(
        const std::vector<const cuMat*>& c_lpmOutputGrads,
        const std::vector<std::shared_ptr<Tensor>>& c_spmInputs,
        const std::vector<std::shared_ptr<Tensor>>& c_spmOutputs
    ) =0;
    virtual std::vector<std::shared_ptr<Tensor>> forward(
        const std::vector<std::shared_ptr<Tensor>>& c_spmInputs
    ) =0;

    std::vector<std::shared_ptr<Tensor>> apply(
        const std::vector<std::shared_ptr<Tensor>>& c_spmInputs
    );
    std::shared_ptr<Tensor> operator()(
        const std::vector<std::shared_ptr<Tensor>>& c_spmInputs
    );
};

// --------------------------
// Context
// --------------------------
class Context{
public:
    Context();
    virtual ~Context();

public:
    Function* _lpFunc;

    std::vector<std::shared_ptr<Tensor>> _spmInputs;
    std::vector<std::weak_ptr<Tensor>> _wpmOutputs;
};

