#include "cuda_function_Add.h"
#include "cuda_tensor.h"
#include <stdexcept>

// --------------------------
// Add
// --------------------------
Add::Add()
    :Function()
{

}
Add::~Add()
{

}

void Add::backward(
    const std::vector<const cufMat*>& c_lpmOutputGrads,
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs,
    const std::vector<std::shared_ptr<Tensor>>& c_spmOutputs
)
{
    // Validate that there are two same-shaped inputs and one output
    // gradient, then accumulate that gradient into both input gradients.
    (void)c_spmOutputs;
    // 
    if( (c_spmInputs.size()!=2)||
        (c_spmInputs[0]==nullptr)||
        (c_spmInputs[1]==nullptr)||
        (c_spmInputs[0]->_mData.shape()!=c_spmInputs[1]->_mData.shape()) )
    {
        throw std::invalid_argument(
            "Add::backward requires exactly two non-null input tensors"
        );
    }
    //
    for( std::size_t nInput=0;nInput<c_spmInputs.size();++nInput )
    {
        if( c_spmInputs[nInput]==nullptr )  {continue;}
        //
        auto& mGrad =c_spmInputs[nInput]->_mGrad;
        if( !mGrad.isContiguous() )
        {
            throw std::invalid_argument(
                "Add::backward: input gradient must be contiguous"
            );
        }
        //
        cuda_axpy(
            mGrad,
            1.0f,
            requireSingleOutputGrad(c_lpmOutputGrads,"Add::backward")
        );
    }
}
std::vector<std::shared_ptr<Tensor>>
Add::forward(
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs
)
{
    // Validate that exactly two same-shaped tensors were supplied,
    // allocate the output tensor, and compute their element-wise sum on CUDA.
    if( (c_spmInputs.size()!=2)||
        (c_spmInputs[0]==nullptr)||
        (c_spmInputs[1]==nullptr)||
        (c_spmInputs[0]->_mData.shape()!=c_spmInputs[1]->_mData.shape()) )
    {
        throw std::invalid_argument(
            "Add::forward requires exactly two non-null input tensors"
        );
    }
    //
    auto spmOutput = std::make_shared<Tensor>( c_spmInputs[0]->_mData.shape() );
    //
    cuda_geam(
        spmOutput->_mData,
        1.0f,
        c_spmInputs[0]->_mData,
        1.0f,
        c_spmInputs[1]->_mData
    );
    //
    return { spmOutput };
}