#include "cuda_tensor.h"
#include "optimizer_SGD.h"


// --------------------------
// SGDParams
// --------------------------
SGDParams::SGDParams()
{

}
SGDParams::~SGDParams()
{

}

// --------------------------
// SGD
// --------------------------
SGD::SGD(Model* lpModel,float fLearningRate)
    :Optimizer( lpModel,fLearningRate )
{

}
SGD::~SGD()
{

}
std::shared_ptr<OptimizerParams>
SGD::createOptimizerParams(Tensor* lpTensor)
{
    (void)lpTensor;
    //
    return std::make_shared<SGDParams>();
}
void SGD::update_param(Tensor* lpTensor,OptimizerParams* lpOptimizerParams)
{
    (void)lpOptimizerParams;
    // W = W - LearingRate * grad
    cuda_axpy(
        lpTensor->_mData,
        -_fLearningRate,
        lpTensor->_mGrad
    );
}

