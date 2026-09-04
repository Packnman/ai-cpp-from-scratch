#include "cuda_tensor.h"
#include "cuda_function_Dropout.h"


// --------------------------
// Dropout
// --------------------------
Dropout::Dropout(float fDropProbability,std::uint64_t nSeed)
    :_fDropProbability( fDropProbability ),
     _crnGenerator( nullptr )
{
    if( (fDropProbability<0.0f)||(fDropProbability>=1.0f) )
    {
        throw std::invalid_argument("Dropout: drop probability must be in [0, 1)");
    }
    checkCurand(curandCreateGenerator(&_crnGenerator,CURAND_RNG_PSEUDO_DEFAULT),"Dropout");
    try
    {
        checkCurand(curandSetPseudoRandomGeneratorSeed(_crnGenerator,nSeed),"Dropout");
    }
    catch( ... )
    {
        curandDestroyGenerator( _crnGenerator );
        _crnGenerator =nullptr;
        throw;
    }
}
Dropout::~Dropout()
{
    if( _crnGenerator!=nullptr )
    {
        curandDestroyGenerator( _crnGenerator );
    }
}
void Dropout::backward(
    const std::vector<const cuMat*>& c_lpmOutputGrads,
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs,
    const std::vector<std::shared_ptr<Tensor>>& c_spmOutputs
)
{
    (void)c_spmOutputs;
    //
    if( c_spmInputs.size()!=1 )
    {
        throw std::runtime_error("Dropout::backward: Dropout requires exactly one input");
    }
    //
    cuda_Dropout_backward(
        c_spmInputs[0]->_mGrad,
        requireSingleOutputGrad(c_lpmOutputGrads,"Dropout::backward"),
        _mMask
    );
}
std::vector<std::shared_ptr<Tensor>>
Dropout::forward(
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs
)
{
    if( (c_spmInputs.size()!=1)||(c_spmInputs[0]==nullptr) )
    {
        throw std::runtime_error("Dropout::forward: Dropout requires exactly one input");
    }
    const int nRows =c_spmInputs[0]->_mData._nRows;
    const int nCols =c_spmInputs[0]->_mData._nCols;
    //
    _mMask =cuMat( nRows,nCols );
    const std::size_t nSize =static_cast<std::size_t>(nRows)*static_cast<std::size_t>(nCols);
    if( _fDropProbability==0.0f )
    {
        cuda_fill( _mMask,1.0f );
    }
    else if( nSize>0 )
    {
        checkCurand(curandGenerateUniform(_crnGenerator,_mMask._lpfDevice,nSize),"Dropout::forward");
    }
    //
    auto spmResult  =std::make_shared<Tensor>( nRows,nCols );
    //
    cuda_Dropout_forward(
        spmResult->_mData,
        c_spmInputs[0]->_mData,_mMask,
        _fDropProbability
    );
    //
    return {spmResult};
}
