#include <cstdio>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "matrix.h"
#include "neuralnet_cifar10.h"

namespace
{
void require(bool isCondition,const std::string& c_strMessage)
{
    if( !isCondition )
    {
        throw std::runtime_error( c_strMessage );
    }
}

void requireSameState(
    const StateDict& c_nmtExpected,
    const StateDict& c_nmtActual
)
{
    require(
        c_nmtExpected.size()==c_nmtActual.size(),
        "state size mismatch"
    );
    for( std::size_t nEntry=0;nEntry<c_nmtExpected.size();++nEntry )
    {
        require(
            c_nmtExpected[nEntry].strName==c_nmtActual[nEntry].strName,
            "state name mismatch"
        );
        Mat mExpected(
            c_nmtExpected[nEntry].lpTensor->_mData._nRows,
            c_nmtExpected[nEntry].lpTensor->_mData._nCols
        );
        Mat mActual(
            c_nmtActual[nEntry].lpTensor->_mData._nRows,
            c_nmtActual[nEntry].lpTensor->_mData._nCols
        );
        c_nmtExpected[nEntry].lpTensor->_mData.upload( mExpected );
        c_nmtActual[nEntry].lpTensor->_mData.upload( mActual );
        for( int nValue=0;nValue<mExpected._nRows*mExpected._nCols;++nValue )
        {
            require(
                mExpected._lpfHost[nValue]==mActual._lpfHost[nValue],
                "loaded state value mismatch"
            );
        }
    }
}

std::shared_ptr<Tensor> makeInput(int nBatch)
{
    auto spmInput =std::make_shared<Tensor>( 3072,nBatch );
    Mat mHost( 3072,nBatch );
    for( int nValue=0;nValue<3072*nBatch;++nValue )
    {
        mHost._lpfHost[nValue] =
            static_cast<float>((nValue%23)-11)/11.0f;
    }
    spmInput->_mData.download( mHost );
    return spmInput;
}

void checkIntermediateShapes()
{
    std::mt19937 rngRandom( 3 );
    Cifar10ConvBlock lyrFirst( 3,32,32,32 );
    lyrFirst.init( rngRandom );
    std::vector<std::shared_ptr<Tensor>> spmValues{makeInput(1)};
    auto spmFirst =lyrFirst.forward( spmValues );
    require(
        (spmFirst->_mData._nRows==32*16*16)&&
        (spmFirst->_mData._nCols==1)&&
        (lyrFirst.outputHeight()==16),
        "first block shape mismatch"
    );

    Cifar10ConvBlock lyrSecond( 32,64,16,16 );
    lyrSecond.init( rngRandom );
    spmValues ={spmFirst};
    auto spmSecond =lyrSecond.forward( spmValues );
    require(
        (spmSecond->_mData._nRows==64*8*8)&&
        (spmSecond->_mData._nCols==1)&&
        (lyrSecond.outputWidth()==8),
        "second block shape mismatch"
    );
}

void checkModel()
{
    constexpr const char* FILE_NAME ="cifar10_model_check.bin";
    NeuralNet_Cifar10 nntModel( 7,0.2f );
    auto spmInput =makeInput( 2 );
    auto spmTarget =std::make_shared<Tensor>( 10,2 );
    Mat mTarget( 10,2 );
    for( int nValue=0;nValue<20;++nValue )
    {
        mTarget._lpfHost[nValue] =0.0f;
    }
    mTarget(1,0) =1.0f;
    mTarget(7,1) =1.0f;
    spmTarget->_mData.download( mTarget );

    std::vector<std::shared_ptr<Tensor>> spmInputs{spmInput};
    auto spmLogits =nntModel.forward( spmInputs );
    require(
        (spmLogits->_mData._nRows==10)&&
        (spmLogits->_mData._nCols==2),
        "logits shape mismatch"
    );

    auto spmLoss =nntModel.loss( spmInput,spmTarget );
    require(
        (spmLoss->_mData._nRows==1)&&(spmLoss->_mData._nCols==1),
        "loss shape mismatch"
    );
    nntModel.zero_grads();
    spmLoss->backward();

    nntModel.setTraining( false );
    require( !nntModel.isTraining(),"evaluation mode was not set" );
    std::vector<std::shared_ptr<Tensor>> spmInputs{spmInput};
    auto spmFirst =nntModel.forward( spmInputs );
    spmInputs ={spmInput};
    auto spmSecond =nntModel.forward( spmInputs );
    Mat mFirst( 10,2 );
    Mat mSecond( 10,2 );
    spmFirst->_mData.upload( mFirst );
    spmSecond->_mData.upload( mSecond );
    for( int nValue=0;nValue<20;++nValue )
    {
        require(
            mFirst._lpfHost[nValue]==mSecond._lpfHost[nValue],
            "evaluation output changed"
        );
    }

    const StateDict nmtState =nntModel.stateDict();
    require( nmtState.size()==12,"unexpected state entry count" );
    nntModel.save( FILE_NAME );
    NeuralNet_Cifar10 nntLoaded( 19 );
    nntLoaded.load( FILE_NAME );
    requireSameState( nmtState,nntLoaded.stateDict() );
    std::remove( FILE_NAME );

    bool isInputRejected =false;
    bool isTargetRejected =false;
    try
    {
        std::vector<std::shared_ptr<Tensor>> spmBad{std::make_shared<Tensor>(3071,1)};
        nntModel.forward( spmBad );
    }
    catch( const std::runtime_error& )
    {
        isInputRejected =true;
    }
    try
    {
        nntModel.loss( spmInput,std::make_shared<Tensor>(9,2) );
    }
    catch( const std::runtime_error& )
    {
        isTargetRejected =true;
    }
    require(
        isInputRejected&&isTargetRejected,
        "model shape validation failed"
    );
}
}

int main()
{
    try
    {
        checkIntermediateShapes();
        checkModel();
        std::cout << "cifar10 model check passed\n";
        return 0;
    }
    catch( const std::exception& c_excError )
    {
        std::remove( "cifar10_model_check.bin" );
        std::cerr << c_excError.what() << '\n';
        return 1;
    }
}
