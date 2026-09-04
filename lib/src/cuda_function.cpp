#include <iostream>
#include <stdio.h>
#include <string>
#include "cuda_matrix.h"
#include "cuda_tensor.h"
#include "cuda_function.h"

// --------------------------
// Function
// --------------------------
Function::Function()
{

}
Function::~Function()
{

}
std::vector<std::shared_ptr<Tensor>>
Function::apply(
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs
)
{
    // 1. 実際の順伝播
    std::vector<std::shared_ptr<Tensor>> spmOutputs =forward( c_spmInputs );
    if( spmOutputs.empty() )
    {
        throw std::runtime_error(
            "Function::apply: forward must return at least one output"
        );
    }

    // 2. この演算を記録するContextを作成
    auto spContext  =std::make_shared<Context>();
    spContext->_lpFunc      =this;
    spContext->_spmInputs   =c_spmInputs;

    // 3. 出力TensorとContextを接続
    for( const auto& c_spmOutput : spmOutputs )
    {
        if( c_spmOutput==nullptr )
        {
            throw std::runtime_error(
                "Function::apply: forward returned a null output"
            );
        }
        spContext->_wpmOutputs.push_back( c_spmOutput );
        c_spmOutput->_spContext =spContext;
    }

    return spmOutputs;
}

std::shared_ptr<Tensor>
Function::operator()(
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs
)
{
    std::vector<std::shared_ptr<Tensor>> spmOutputs =apply( c_spmInputs );
    if( spmOutputs.size()!=1 )
    {
        throw std::runtime_error(
            "Function::operator(): use apply() for a multi-output Function"
        );
    }

    return spmOutputs[0];
}

void Function::checkCurand(curandStatus_t crnStatus,const char* c_lpszOperation)
{
    if( crnStatus!=CURAND_STATUS_SUCCESS )
    {
        throw std::runtime_error(
            std::string(c_lpszOperation)+": cuRAND failed with status "+
            std::to_string(static_cast<int>(crnStatus))
        );
    }
}

const cuMat& Function::requireSingleOutputGrad(
    const std::vector<const cuMat*>& c_lpmOutputGrads,
    const char* c_lpszFunctionName
)
{
    if( (c_lpmOutputGrads.size()!=1)||(c_lpmOutputGrads[0]==nullptr) )
    {
        throw std::runtime_error(
            std::string(c_lpszFunctionName)+": exactly one output gradient is required"
        );
    }
    return *c_lpmOutputGrads[0];
}
const cuMat& Function::singleGrad(
    const std::vector<const cuMat*>& c_lpmOutputGrads,
    const char* c_lpszFunctionName
)
{
    if( (c_lpmOutputGrads.size()!=1)||(c_lpmOutputGrads[0]==nullptr) )
    {
        throw std::runtime_error(
            std::string( c_lpszFunctionName ) +
            ": exactly one output gradient is required"
        );
    }
    return *c_lpmOutputGrads[0];
}

// --------------------------
// Context
// --------------------------
Context::Context()
{
    _lpFunc =nullptr;
}
Context::~Context()
{

}

