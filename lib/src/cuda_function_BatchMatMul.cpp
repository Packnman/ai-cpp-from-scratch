#include "cuda_function_BatchMatMul.h"

#include <climits>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cuda_tensor.h"


BatchMatMul::BatchMatMul()
    :Function()
{
}

BatchMatMul::~BatchMatMul()
{
}

BatchMatMul::ShapeInfo BatchMatMul::_validateInputs(
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs,
    const char* c_lpszOperation
)
{
    if( c_spmInputs.size()!=2 )
    {
        throw std::runtime_error(
            std::string(c_lpszOperation)+": exactly two inputs are required"
        );
    }
    if( (c_spmInputs[0]==nullptr)||(c_spmInputs[1]==nullptr) )
    {
        throw std::runtime_error(
            std::string(c_lpszOperation)+": inputs must not be null"
        );
    }

    const cufMat& c_mA =c_spmInputs[0]->_mData;
    const cufMat& c_mB =c_spmInputs[1]->_mData;
    if( (c_mA.dim()<2)||(c_mB.dim()<2) )
    {
        throw std::invalid_argument(
            std::string(c_lpszOperation)+": input rank must be at least two"
        );
    }
    if( c_mA.dim()!=c_mB.dim() )
    {
        throw std::invalid_argument(
            std::string(c_lpszOperation)+": batch shapes must match"
        );
    }
    if( c_mA.size(1)!=c_mB.size(0) )
    {
        throw std::invalid_argument(
            std::string(c_lpszOperation)+": inner dimensions must match"
        );
    }
    for( std::size_t nDimension=2;nDimension<c_mA.dim();++nDimension )
    {
        if( c_mA.size(nDimension)!=c_mB.size(nDimension) )
        {
            throw std::invalid_argument(
                std::string(c_lpszOperation)+": batch shapes must match"
            );
        }
    }
    if( !c_mA.isContiguous()||!c_mB.isContiguous() )
    {
        throw std::invalid_argument(
            std::string(c_lpszOperation)+": contiguous inputs are required"
        );
    }

    const auto requireIntExtent =[c_lpszOperation](std::int64_t nExtent)
    {
        if( nExtent>INT_MAX )
        {
            throw std::overflow_error(
                std::string(c_lpszOperation)+": tensor is too large"
            );
        }
    };
    for( std::int64_t nExtent:c_mA.shape() ) requireIntExtent(nExtent);
    for( std::int64_t nExtent:c_mB.shape() ) requireIntExtent(nExtent);

    std::size_t nBatch =1;
    for( std::size_t nDimension=2;nDimension<c_mA.dim();++nDimension )
    {
        const std::size_t nExtent =static_cast<std::size_t>(
            c_mA.size(nDimension)
        );
        if( (nExtent!=0)&&(nBatch>static_cast<std::size_t>(INT_MAX)/nExtent) )
        {
            throw std::overflow_error(
                std::string(c_lpszOperation)+": batch shape is too large"
            );
        }
        nBatch *=nExtent;
    }

    std::vector<std::int64_t> shapeOutput =c_mA.shape();
    shapeOutput[1] =c_mB.size(1);
    std::size_t nOutputElements =nBatch;
    for( const std::int64_t nExtent:{c_mA.size(0),c_mB.size(1)} )
    {
        if( (nExtent!=0)&&
            (nOutputElements>static_cast<std::size_t>(INT_MAX)/
                             static_cast<std::size_t>(nExtent)) )
        {
            throw std::overflow_error(
                std::string(c_lpszOperation)+": output tensor is too large"
            );
        }
        nOutputElements *=static_cast<std::size_t>(nExtent);
    }
    if( (c_mA.numel()>static_cast<std::size_t>(INT_MAX))||
        (c_mB.numel()>static_cast<std::size_t>(INT_MAX)) )
    {
        throw std::overflow_error(
            std::string(c_lpszOperation)+": input tensor is too large"
        );
    }

    return {
        static_cast<int>(c_mA.size(0)),
        static_cast<int>(c_mA.size(1)),
        static_cast<int>(c_mB.size(1)),
        static_cast<int>(nBatch),
        std::move(shapeOutput)
    };
}

void BatchMatMul::backward(
    const std::vector<const cufMat*>& c_lpmOutputGrads,
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs,
    const std::vector<std::shared_ptr<Tensor>>& c_spmOutputs
)
{
    (void)c_spmOutputs;
    //
    const ShapeInfo shape =_validateInputs(
        c_spmInputs,"BatchMatMul::backward"
    );
    const cufMat& c_mOutputGrad =requireSingleOutputGrad(
        c_lpmOutputGrads,"BatchMatMul::backward"
    );
    if( c_mOutputGrad.shape()!=shape.shapeOutput )
    {
        throw std::invalid_argument(
            "BatchMatMul::backward: output gradient shape mismatch"
        );
    }
    if( !c_mOutputGrad.isContiguous() )
    {
        throw std::invalid_argument(
            "BatchMatMul::backward: contiguous output gradient required"
        );
    }

    cuda_BatchMatMul_backward(
        c_spmInputs[0]->_mGrad,
        c_spmInputs[1]->_mGrad,
        c_mOutputGrad,
        c_spmInputs[0]->_mData,
        c_spmInputs[1]->_mData,
        shape.nM,
        shape.nK,
        shape.nN,
        shape.nBatch
    );
}

std::vector<std::shared_ptr<Tensor>>
BatchMatMul::forward(
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs
)
{
    const ShapeInfo shape =_validateInputs(
        c_spmInputs,"BatchMatMul::forward"
    );
    auto spmResult =std::make_shared<Tensor>(shape.shapeOutput);
    cuda_BatchMatMul_forward(
        spmResult->_mData,
        c_spmInputs[0]->_mData,
        c_spmInputs[1]->_mData,
        shape.nM,
        shape.nK,
        shape.nN,
        shape.nBatch
    );
    //
    return {spmResult};
}
