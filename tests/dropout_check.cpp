#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "cuda_function_Dropout.h"
#include "cuda_tensor.h"
#include "matrix.h"
#include "neuralnet_mnist.h"

namespace {
void require(bool condition,const std::string& message)
{
    if( !condition ) {throw std::runtime_error(message);}
}

Mat upload(const cuMat& value)
{
    Mat host( value._nRows,value._nCols );
    value.upload( host );
    return host;
}

std::shared_ptr<Tensor> makeInput(int size)
{
    auto input =std::make_shared<Tensor>( size,1 );
    Mat host( size,1 );
    for( int i=0;i<size;++i ) {host(i,0)=static_cast<float>(i+1);}
    input->_mData.download( host );
    return input;
}

void testProbabilityValidation()
{
    bool rejectedNegative =false;
    bool rejectedOne =false;
    try {Dropout invalid(-0.01f,1);}
    catch( const std::invalid_argument& ) {rejectedNegative=true;}
    try {Dropout invalid(1.0f,1);}
    catch( const std::invalid_argument& ) {rejectedOne=true;}
    require( rejectedNegative,"negative probability was accepted" );
    require( rejectedOne,"probability one was accepted" );

    auto input =makeInput( 32 );
    Dropout identity( 0.0f,1 );
    auto output =identity( {input} );
    Mat inputHost =upload( input->_mData );
    Mat outputHost =upload( output->_mData );
    for( int i=0;i<32;++i )
    {
        require( inputHost(i,0)==outputHost(i,0),"p=0 is not identity" );
    }
    output->backward();
    Mat gradHost =upload( input->_mGrad );
    for( int i=0;i<32;++i )
    {
        require( gradHost(i,0)==1.0f,"p=0 backward is not identity" );
    }
}

void testMaskAndBackward()
{
    constexpr int SIZE =4096;
    auto firstInput =makeInput( SIZE );
    auto secondInput =makeInput( SIZE );
    Dropout first( 0.2f,12345 );
    Dropout second( 0.2f,12345 );
    auto firstOutput =first( {firstInput} );
    auto secondOutput =second( {secondInput} );
    Mat firstHost =upload( firstOutput->_mData );
    Mat secondHost =upload( secondOutput->_mData );

    int dropped =0;
    int kept =0;
    for( int i=0;i<SIZE;++i )
    {
        require( firstHost(i,0)==secondHost(i,0),"same seed gave different mask" );
        const float expected =static_cast<float>(i+1)/0.8f;
        if( firstHost(i,0)==0.0f ) {++dropped;}
        else
        {
            require(
                std::fabs(firstHost(i,0)-expected)<=expected*1.0e-6f,
                "kept output has wrong scale"
            );
            ++kept;
        }
    }
    require( dropped>0,"mask dropped no values" );
    require( kept>0,"mask kept no values" );

    firstOutput->backward();
    Mat gradHost =upload( firstInput->_mGrad );
    for( int i=0;i<SIZE;++i )
    {
        const float expected =firstHost(i,0)==0.0f ? 0.0f : 1.25f;
        require( gradHost(i,0)==expected,"backward did not reuse mask" );
    }
}

void testEvaluationBypassesDropout()
{
    MnistNeuralNet withDropout( 77,0.2f );
    MnistNeuralNet withoutDropout( 77,0.0f );
    withDropout.setTraining( false );
    withoutDropout.setTraining( false );
    require( !withDropout.isTraining(),"evaluation mode was not set" );

    auto input =std::make_shared<Tensor>( 784,2 );
    Mat host( 784,2 );
    for( int i=0;i<784*2;++i )
    {
        host._lpfHost[i] =static_cast<float>((i%17)-8)/8.0f;
    }
    input->_mData.download( host );
    std::vector<std::shared_ptr<Tensor>> args{input};
    auto first =withDropout.forward( args );
    auto second =withDropout.forward( args );
    auto baseline =withoutDropout.forward( args );
    Mat firstHost =upload( first->_mData );
    Mat secondHost =upload( second->_mData );
    Mat baselineHost =upload( baseline->_mData );
    for( int i=0;i<firstHost._nRows*firstHost._nCols;++i )
    {
        require(
            firstHost._lpfHost[i]==secondHost._lpfHost[i],
            "evaluation output changed"
        );
        require(
            firstHost._lpfHost[i]==baselineHost._lpfHost[i],
            "evaluation did not bypass dropout"
        );
    }

    withDropout.setTraining( true );
    require( withDropout.isTraining(),"training mode was not restored" );
}
}

int main()
{
    try
    {
        testProbabilityValidation();
        testMaskAndBackward();
        testEvaluationBypassesDropout();
        std::cout << "dropout check passed\n";
        return 0;
    }
    catch( const std::exception& error )
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
