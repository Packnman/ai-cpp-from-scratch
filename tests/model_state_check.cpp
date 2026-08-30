#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "matrix.h"
#include "neuralnet.h"

namespace {
    void require(bool condition,const std::string& message)
    {
        if( !condition )
        {
            throw std::runtime_error( message );
        }
    }

    void requireSameTensor(const Tensor& expected,const Tensor& actual)
    {
        require(
            (expected._mData._nRows==actual._mData._nRows)&&
            (expected._mData._nCols==actual._mData._nCols),
            "tensor shape mismatch"
        );

        Mat expectedHost( expected._mData._nRows,expected._mData._nCols );
        Mat actualHost( actual._mData._nRows,actual._mData._nCols );
        expected._mData.upload( expectedHost );
        actual._mData.upload( actualHost );

        const int size =expectedHost._nRows*expectedHost._nCols;
        for( int i=0;i<size;++i )
        {
            require(
                expectedHost._lpfHost[i]==actualHost._lpfHost[i],
                "tensor value mismatch"
            );
        }
    }
}

int main()
{
    constexpr const char* FILE_NAME ="model_state_check.bin";

    try
    {
        NeuralNet source( 1 );
        NeuralNet destination( 2 );

        const std::vector<std::string> expectedNames{
            "hidden1.weight",
            "hidden1.bias",
            "hidden2.weight",
            "hidden2.bias",
            "output.weight",
            "output.bias"
        };
        const std::vector<NamedTensor> named =source.namedParameters();
        require( named.size()==expectedNames.size(),"parameter count mismatch" );
        for( std::size_t i=0;i<named.size();++i )
        {
            require( named[i].strName==expectedNames[i],"parameter name mismatch" );
        }
        require( source.getParams().size()==named.size(),"getParams count mismatch" );
        require( source.stateDict().size()==named.size(),"stateDict count mismatch" );

        source.save( FILE_NAME );
        destination.load( FILE_NAME );

        const StateDict sourceState =source.stateDict();
        const StateDict destinationState =destination.stateDict();
        require( sourceState.size()==destinationState.size(),"loaded state count mismatch" );
        for( std::size_t i=0;i<sourceState.size();++i )
        {
            require( sourceState[i].strName==destinationState[i].strName,"loaded state name mismatch" );
            requireSameTensor( *sourceState[i].lpTensor,*destinationState[i].lpTensor );
        }

        std::remove( FILE_NAME );
        std::cout << "model state check passed\n";
        return 0;
    }
    catch( const std::exception& error )
    {
        std::remove( FILE_NAME );
        std::cerr << error.what() << '\n';
        return 1;
    }
}
