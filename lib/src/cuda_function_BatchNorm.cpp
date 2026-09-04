#include "cuda_tensor.h"
#include "cuda_function_BatchNorm.h"


// --------------------------
// BatchNorm
// --------------------------
BatchNorm::BatchNorm(
    Tensor* lpGamma,
    Tensor* lpBeta,
    Tensor* lpRunningMean,
    Tensor* lpRunningVar,
    float fMomentum,
    float fEpsilon
)
    :_lpmGamma( lpGamma ),
     _lpmBeta( lpBeta ),
     _lpmRunningMean( lpRunningMean ),
     _lpmRunningVar( lpRunningVar ),
     _fMomentum( fMomentum ),
     _fEpsilon( fEpsilon ),
     _isTraining( true ),
     _wasTraining( true )
{
    if( (lpGamma==nullptr)||(lpBeta==nullptr)||
        (lpRunningMean==nullptr)||(lpRunningVar==nullptr) )
    {
        throw std::invalid_argument("BatchNorm: tensors must not be null");
    }
    if( !(fMomentum>=0.0f&&fMomentum<=1.0f) )
    {
        throw std::invalid_argument("BatchNorm: momentum must be in [0, 1]");
    }
    if( !(fEpsilon>0.0f) )
    {
        throw std::invalid_argument("BatchNorm: epsilon must be positive");
    }
    const int nFeatures =lpGamma->_mData._nRows;
    if( (lpGamma->_mData._nCols!=1)||
        (lpBeta->_mData._nRows!=nFeatures)||(lpBeta->_mData._nCols!=1)||
        (lpRunningMean->_mData._nRows!=nFeatures)||(lpRunningMean->_mData._nCols!=1)||
        (lpRunningVar->_mData._nRows!=nFeatures)||(lpRunningVar->_mData._nCols!=1) )
    {
        throw std::invalid_argument("BatchNorm: state tensors must have shape features x 1");
    }
}
BatchNorm::~BatchNorm()
{
}
void BatchNorm::setTraining(bool isTraining)
{
    _isTraining =isTraining;
}
bool BatchNorm::isTraining() const
{
    return _isTraining;
}
void BatchNorm::backward(
    const std::vector<const cuMat*>& c_lpmOutputGrads,
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs,
    const std::vector<std::shared_ptr<Tensor>>& c_spmOutputs
)
{
    (void)c_spmOutputs;
    //
    if( c_spmInputs.size()!=1 )
    {
        throw std::runtime_error("BatchNorm::backward: exactly one input is required");
    }
    //
    cuda_BatchNorm_backward(
        c_spmInputs[0]->_mGrad,
        _lpmGamma->_mGrad,
        _lpmBeta->_mGrad,
        requireSingleOutputGrad(c_lpmOutputGrads,"BatchNorm::backward"),
        _lpmGamma->_mData,
        _mNormalized,_mInvStd,
        _wasTraining
    );
}
std::vector<std::shared_ptr<Tensor>>
BatchNorm::forward(
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs
)
{
    if( (c_spmInputs.size()!=1)||(c_spmInputs[0]==nullptr) )
    {
        throw std::runtime_error("BatchNorm::forward: exactly one input is required");
    }

    const int nFeatures =_lpmGamma->_mData._nRows;
    const int nBatch    =c_spmInputs[0]->_mData._nCols;
    if( (c_spmInputs[0]->_mData._nRows!=nFeatures)||
        (c_spmInputs[0]->_mData._nCols<=0) )
    {
        throw std::runtime_error("BatchNorm::forward: input must have shape features x batch");
    }

    _mNormalized    =cuMat( nFeatures,nBatch );
    _mInvStd        =cuMat( nFeatures,1 );
    auto spmResult  =std::make_shared<Tensor>( nFeatures,nBatch );
    _wasTraining    =_isTraining;
    //
    if( _isTraining )
    {
        cuda_BatchNorm_forward_training(
            spmResult->_mData,
            c_spmInputs[0]->_mData,
            _lpmGamma->_mData,
            _lpmBeta->_mData,
            _lpmRunningMean->_mData,
            _lpmRunningVar->_mData,
            _mNormalized,
            _mInvStd,
            _fMomentum,
            _fEpsilon
        );
    }
    else
    {
        cuda_BatchNorm_forward_evaluation(
            spmResult->_mData,
            c_spmInputs[0]->_mData,
            _lpmGamma->_mData,
            _lpmBeta->_mData,
            _lpmRunningMean->_mData,
            _lpmRunningVar->_mData,
            _mNormalized,
            _mInvStd,
            _fEpsilon
        );
    }
    return {spmResult};
}
