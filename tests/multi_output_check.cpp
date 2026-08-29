#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "cuda_function.h"
#include "cuda_tensor.h"
#include "matrix.h"

namespace {
    void require(bool condition,const std::string& message)
    {
        if( !condition )
        {
            throw std::runtime_error( message );
        }
    }

    class Duplicate final : public Function
    {
    public:
        TensorList forward(const TensorList& inputs) override
        {
            require( inputs.size()==1,"Duplicate requires one input" );

            auto first =std::make_shared<Tensor>(
                inputs[0]->_mData._nRows,
                inputs[0]->_mData._nCols
            );
            auto second =std::make_shared<Tensor>(
                inputs[0]->_mData._nRows,
                inputs[0]->_mData._nCols
            );
            cuda_geam( first->_mData,1.0f,inputs[0]->_mData,0.0f,inputs[0]->_mData );
            cuda_geam( second->_mData,1.0f,inputs[0]->_mData,0.0f,inputs[0]->_mData );

            return {first,second};
        }

        void backward(
            const TensorGradList& outputGrads,
            const TensorList& inputs,
            const TensorList& outputs
        ) override
        {
            (void)outputs;
            require( inputs.size()==1,"Duplicate backward input mismatch" );
            require( outputGrads.size()==2,"Duplicate backward output mismatch" );

            for( const cuMat* grad : outputGrads )
            {
                if( grad!=nullptr )
                {
                    cuda_axpy( inputs[0]->_mGrad,1.0f,*grad );
                }
            }
        }
    };

    class Add final : public Function
    {
    public:
        TensorList forward(const TensorList& inputs) override
        {
            require( inputs.size()==2,"Add requires two inputs" );
            require(
                (inputs[0]->_mData._nRows==inputs[1]->_mData._nRows)&&
                (inputs[0]->_mData._nCols==inputs[1]->_mData._nCols),
                "Add input shape mismatch"
            );

            auto output =std::make_shared<Tensor>(
                inputs[0]->_mData._nRows,
                inputs[0]->_mData._nCols
            );
            cuda_geam(
                output->_mData,
                1.0f,
                inputs[0]->_mData,
                1.0f,
                inputs[1]->_mData
            );
            return {output};
        }

        void backward(
            const TensorGradList& outputGrads,
            const TensorList& inputs,
            const TensorList& outputs
        ) override
        {
            (void)outputs;
            require( inputs.size()==2,"Add backward input mismatch" );
            require(
                (outputGrads.size()==1)&&(outputGrads[0]!=nullptr),
                "Add backward output mismatch"
            );

            cuda_axpy( inputs[0]->_mGrad,1.0f,*outputGrads[0] );
            cuda_axpy( inputs[1]->_mGrad,1.0f,*outputGrads[0] );
        }
    };

    float readScalar(const cuMat& value)
    {
        Mat host( 1,1 );
        value.upload( host );
        return host( 0,0 );
    }
}

int main()
{
    try
    {
        auto input =std::make_shared<Tensor>( 1,1 );
        Mat inputHost( 1,1 );
        inputHost( 0,0 ) =3.0f;
        input->_mData.download( inputHost );

        Duplicate duplicate;
        Add add;

        TensorList branches =duplicate.apply({input});
        require( branches.size()==2,"Duplicate must return two outputs" );
        require(
            branches[0]->_spContext==branches[1]->_spContext,
            "multi-output tensors must share one Context"
        );

        TensorPtr result =add( branches );
        require( readScalar(result->_mData)==6.0f,"forward result mismatch" );

        result->backward();
        require( readScalar(input->_mGrad)==2.0f,"multi-output gradient mismatch" );

        std::cout << "multi-output check passed\n";
        return 0;
    }
    catch( const std::exception& error )
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
