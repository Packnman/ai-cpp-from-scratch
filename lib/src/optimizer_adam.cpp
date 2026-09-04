#include <math.h>
#include "cuda_tensor.h"
#include "optimizer_adam.h"


// --------------------------
// AdamParams
// --------------------------
AdamParams::AdamParams(int nRows,int nCols)
    :_mM( nRows,nCols ),
     _mV( nRows,nCols )
{
    cuda_fill( _mM,0.0f );
    cuda_fill( _mV,0.0f );
}
AdamParams::~AdamParams()
{

}

// --------------------------
// Adam
// --------------------------
Adam::Adam(Model* lpModel,float fLearningRate)
    :Optimizer(lpModel,fLearningRate)
{

}
Adam::~Adam()
{

}
std::shared_ptr<OptimizerParams>
Adam::createOptimizerParams(Tensor* lpTensor)
{
    return std::make_shared<AdamParams>(
        lpTensor->_mData._nRows,
        lpTensor->_mData._nCols
    );
}
void Adam::update_param(Tensor* lpTensor,OptimizerParams* lpOptimizerParams)
{
    // 更新手順はREADME参照
    AdamParams* lpAdamParams    =dynamic_cast<AdamParams*>(lpOptimizerParams);

    if( lpAdamParams==nullptr )
    {
        throw std::runtime_error(
            "Adam::update_param: invalid optimizer params"
        );
    }

    const float c_fBeta1      =0.9f;
    const float c_fBeta2      =0.999f;
    const float c_fEpsilon    =1.0e-8f;

    float fBeta1Correction =1.0f -powf(
        c_fBeta1,
        static_cast<float>(_nStep)
    );
    float fBeta2Correction =1.0f -powf(
        c_fBeta2,
        static_cast<float>(_nStep)
    );

    cuda_Adam_update(
        lpTensor->_mData,
        lpTensor->_mGrad,
        lpAdamParams->_mM,
        lpAdamParams->_mV,
        _fLearningRate,
        c_fBeta1,
        c_fBeta2,
        fBeta1Correction,
        fBeta2Correction,
        c_fEpsilon
    );
}
