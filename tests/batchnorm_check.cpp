#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_function.h"
#include "cuda_tensor.h"
#include "matrix.h"

namespace {
void require(bool condition,const std::string& message)
{
    if( !condition ) {throw std::runtime_error(message);}
}

void requireNear(float actual,float expected,float tolerance,const std::string& message)
{
    if( std::fabs(actual-expected)>tolerance )
    {
        throw std::runtime_error(message);
    }
}

Mat upload(const cuMat& value)
{
    Mat host( value._nRows,value._nCols );
    value.upload( host );
    return host;
}

void fillTensor(Tensor& tensor,const std::vector<float>& values)
{
    require(
        values.size()==static_cast<std::size_t>(tensor._mData._nRows*tensor._mData._nCols),
        "test data size mismatch"
    );
    Mat host( tensor._mData._nRows,tensor._mData._nCols );
    for( std::size_t i=0;i<values.size();++i ) {host._lpfHost[i]=values[i];}
    tensor._mData.download( host );
}

void testValidation()
{
    Tensor value( 2,1 );
    bool rejectedMomentum =false;
    bool rejectedEpsilon =false;
    try {BatchNorm invalid(&value,&value,&value,&value,1.1f,1.0e-5f);}
    catch( const std::invalid_argument& ) {rejectedMomentum=true;}
    try {BatchNorm invalid(&value,&value,&value,&value,0.1f,0.0f);}
    catch( const std::invalid_argument& ) {rejectedEpsilon=true;}
    require( rejectedMomentum,"invalid momentum was accepted" );
    require( rejectedEpsilon,"invalid epsilon was accepted" );
}

void testForwardBackwardAndEvaluation()
{
    constexpr int FEATURES =2;
    constexpr int BATCH =3;
    constexpr float EPSILON =1.0e-5f;

    Tensor gamma( FEATURES,1 );
    Tensor beta( FEATURES,1 );
    Tensor runningMean( FEATURES,1 );
    Tensor runningVar( FEATURES,1 );
    fillTensor( gamma,{2.0f,0.5f} );
    fillTensor( beta,{-1.0f,1.0f} );
    fillTensor( runningMean,{0.0f,0.0f} );
    fillTensor( runningVar,{1.0f,1.0f} );

    auto input =std::make_shared<Tensor>( FEATURES,BATCH );
    const std::vector<float> values{1.0f,2.0f,2.0f,4.0f,3.0f,6.0f};
    fillTensor( *input,values );

    BatchNorm batchNorm( &gamma,&beta,&runningMean,&runningVar,0.1f,EPSILON );
    auto output =batchNorm( {input} );
    Mat outputHost =upload( output->_mData );

    const float means[FEATURES]{2.0f,4.0f};
    const float variances[FEATURES]{2.0f/3.0f,8.0f/3.0f};
    const float gammas[FEATURES]{2.0f,0.5f};
    const float betas[FEATURES]{-1.0f,1.0f};
    for( int batch=0;batch<BATCH;++batch )
    {
        for( int feature=0;feature<FEATURES;++feature )
        {
            const int index =batch*FEATURES+feature;
            const float normalized =(values[index]-means[feature])/
                std::sqrt(variances[feature]+EPSILON);
            requireNear(
                outputHost._lpfHost[index],
                gammas[feature]*normalized+betas[feature],
                2.0e-5f,
                "training forward mismatch"
            );
        }
    }

    Mat runningMeanHost =upload( runningMean._mData );
    Mat runningVarHost =upload( runningVar._mData );
    requireNear( runningMeanHost(0,0),0.2f,1.0e-6f,"running mean mismatch" );
    requireNear( runningMeanHost(1,0),0.4f,1.0e-6f,"running mean mismatch" );
    requireNear( runningVarHost(0,0),1.0f,1.0e-6f,"running variance mismatch" );
    requireNear( runningVarHost(1,0),1.3f,1.0e-6f,"running variance mismatch" );

    cuMat outputGrad( FEATURES,BATCH );
    Mat outputGradHost( FEATURES,BATCH );
    const std::vector<float> gradients{1.0f,-1.0f,2.0f,3.0f,4.0f,2.0f};
    for( std::size_t i=0;i<gradients.size();++i ) {outputGradHost._lpfHost[i]=gradients[i];}
    outputGrad.download( outputGradHost );
    batchNorm.backward( {&outputGrad},{input},{output} );

    Mat inputGradHost =upload( input->_mGrad );
    Mat gammaGradHost =upload( gamma._mGrad );
    Mat betaGradHost =upload( beta._mGrad );
    for( int feature=0;feature<FEATURES;++feature )
    {
        float gradSum =0.0f;
        float gradNormalizedSum =0.0f;
        for( int batch=0;batch<BATCH;++batch )
        {
            const int index =batch*FEATURES+feature;
            const float normalized =(values[index]-means[feature])/
                std::sqrt(variances[feature]+EPSILON);
            gradSum +=gradients[index];
            gradNormalizedSum +=gradients[index]*normalized;
        }
        requireNear( gammaGradHost(feature,0),gradNormalizedSum,2.0e-5f,"gamma gradient mismatch" );
        requireNear( betaGradHost(feature,0),gradSum,2.0e-5f,"beta gradient mismatch" );
        for( int batch=0;batch<BATCH;++batch )
        {
            const int index =batch*FEATURES+feature;
            const float normalized =(values[index]-means[feature])/
                std::sqrt(variances[feature]+EPSILON);
            const float expected =gammas[feature]/
                std::sqrt(variances[feature]+EPSILON)/static_cast<float>(BATCH)*
                (static_cast<float>(BATCH)*gradients[index]-gradSum-
                 normalized*gradNormalizedSum);
            requireNear( inputGradHost._lpfHost[index],expected,4.0e-5f,"input gradient mismatch" );
        }
    }

    batchNorm.setTraining( false );
    require( !batchNorm.isTraining(),"evaluation mode was not set" );
    auto evaluation =batchNorm( {input} );
    Mat evaluationHost =upload( evaluation->_mData );
    for( int batch=0;batch<BATCH;++batch )
    {
        for( int feature=0;feature<FEATURES;++feature )
        {
            const int index =batch*FEATURES+feature;
            const float expected =gammas[feature]*
                (values[index]-runningMeanHost(feature,0))/
                std::sqrt(runningVarHost(feature,0)+EPSILON)+betas[feature];
            requireNear( evaluationHost._lpfHost[index],expected,2.0e-5f,"evaluation forward mismatch" );
        }
    }
    Mat meanAfterEvaluation =upload( runningMean._mData );
    Mat varAfterEvaluation =upload( runningVar._mData );
    for( int feature=0;feature<FEATURES;++feature )
    {
        require(
            meanAfterEvaluation(feature,0)==runningMeanHost(feature,0),
            "evaluation changed running mean"
        );
        require(
            varAfterEvaluation(feature,0)==runningVarHost(feature,0),
            "evaluation changed running variance"
        );
    }
}
}

int main()
{
    try
    {
        testValidation();
        testForwardBackwardAndEvaluation();
        std::cout << "batchnorm check passed\n";
        return 0;
    }
    catch( const std::exception& error )
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
