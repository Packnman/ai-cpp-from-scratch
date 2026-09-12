#pragma once

#include <cstdint>
#include <curand.h>
#include <vector>
#include <memory>
#include "cuda_matrix.h"

class Tensor;
class IndexFunction;

using TensorPtr =std::shared_ptr<Tensor>;
using TensorList =std::vector<TensorPtr>;
using TensorGradList =std::vector<const cufMat*>;

// --------------------------
// Function
// --------------------------
class Function{
public:
    Function();
    virtual ~Function();

protected:
    void checkCurand(curandStatus_t crnStatus,const char* c_lpszOperation);
    const cufMat& requireSingleOutputGrad(
        const std::vector<const cufMat*>& c_lpmOutputGrads,
        const char* c_lpszFunctionName
    );
    const cufMat& singleGrad(
        const std::vector<const cufMat*>& c_lpmOutputGrads,
        const char* c_lpszFunctionName
    );

public:
    virtual void backward(
        const std::vector<const cufMat*>& c_lpmOutputGrads,
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
// IndexFunction
// --------------------------
// Integer tensors are lookup keys, not differentiable Tensor inputs. This
// parallel interface records them in the autograd Context while leaving the
// existing floating-point Function API unchanged.
class IndexFunction{
public:
    IndexFunction() =default;
    virtual ~IndexFunction() =default;

    virtual void backward(
        const TensorGradList& c_lpmOutputGrads,
        const std::shared_ptr<const cunMat>& c_spmIndices,
        const TensorList& c_spmOutputs
    ) =0;
    virtual TensorList forward(
        const std::shared_ptr<const cunMat>& c_spmIndices
    ) =0;

    TensorList apply(const std::shared_ptr<const cunMat>& c_spmIndices);
    TensorPtr operator()(const std::shared_ptr<const cunMat>& c_spmIndices);
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
    IndexFunction* _lpIndexFunc;
    std::shared_ptr<Function> _spFunction; // Graph-local operation ownership

    std::vector<std::shared_ptr<Tensor>> _spmInputs;
    std::shared_ptr<const cunMat> _spmIndices;
    std::vector<std::weak_ptr<Tensor>> _wpmOutputs;
};

#include "cuda_function_Add.h"
#include "cuda_function_BatchMatMul.h"
#include "cuda_function_BatchNorm.h"
#include "cuda_function_Conv2D.h"
#include "cuda_function_Dropout.h"
#include "cuda_function_Embedding.h"
#include "cuda_function_GELU.h"
#include "cuda_function_LayerNorm.h"
#include "cuda_function_Linear.h"
#include "cuda_function_Mask.h"
#include "cuda_function_Permute.h"
#include "cuda_function_Pooling.h"
#include "cuda_function_ReLU.h"
#include "cuda_function_Reshape.h"
#include "cuda_function_Scale.h"
#include "cuda_function_Softmax.h"
#include "cuda_function_SoftmaxCrossEntropy.h"