#include "cuda_tensor.h"
#include "cuda_function_Pooling.h"


Pooling::Pooling(
    int nChannels,
    int nInputHeight,
    int nInputWidth,
    int nKernelSize,
    int nStride
)
    :_nChannels( nChannels ),
     _nInputHeight( nInputHeight ),
     _nInputWidth( nInputWidth ),
     _nKernelSize( nKernelSize ),
     _nStride( nStride ),
     _nOutputHeight( 0 ),
     _nOutputWidth( 0 )
{
    if( (nChannels<=0)||(nInputHeight<=0)||(nInputWidth<=0)||
        (nKernelSize<=0)||(nStride<=0) )
    {
        throw std::invalid_argument(
            "Pooling: invalid channels, image, kernel, or stride"
        );
    }
    if( (nKernelSize>nInputHeight)||(nKernelSize>nInputWidth) )
    {
        throw std::invalid_argument( "Pooling: kernel exceeds input" );
    }
    const long long nInputRows      =static_cast<long long>( nInputHeight ) * nInputWidth * nChannels;
    // H_out = floor((H_in - K) / S) + 1
    // W_out = floor((W_in - K) / S) + 1
    const long long nOutputHeight   =( nInputHeight - nKernelSize ) / nStride + 1;
    const long long nOutputWidth    =( nInputWidth - nKernelSize ) / nStride + 1;
    if( (nInputRows>INT_MAX)||(nOutputHeight*nOutputWidth*nChannels>INT_MAX) )
    {
        throw std::overflow_error(
            "Pooling: dimensions exceed int range"
        );
    }
    _nOutputHeight  =static_cast<int>( nOutputHeight );
    _nOutputWidth   =static_cast<int>( nOutputWidth );
}
Pooling::~Pooling() = default;

std::vector<std::shared_ptr<Tensor>>
Pooling::forward(
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs
)
{
    if( (c_spmInputs.size()!=1)||!c_spmInputs[0] )
    {
        throw std::runtime_error(
            "Pooling::forward: exactly one non-null input is required"
        );
    }
    const int nInputRows    =_nInputHeight * _nInputWidth * _nChannels;
    const int nOutputRows   =_nOutputHeight * _nOutputWidth * _nChannels;
    if( (c_spmInputs[0]->_mData._nRows!=nInputRows)||(c_spmInputs[0]->_mData._nCols<=0) )
    {
        throw std::runtime_error( "Pooling::forward: input shape mismatch" );
    }
    if( static_cast<long long>(nOutputRows)*c_spmInputs[0]->_mData._nCols>INT_MAX )
    {
        throw std::overflow_error(
            "Pooling::forward: output exceeds int range"
        );
    }
    auto spmOutput  =std::make_shared<Tensor>( nOutputRows, c_spmInputs[0]->_mData._nCols );
    // Y[n,y_out,x_out,c]
    //   = max_(0<=k_y,k_x<K) X[n,y_out*S+k_y,x_out*S+k_x,c]
    cuda_MaxPool2D_forward(
        spmOutput->_mData,
        c_spmInputs[0]->_mData,
        _nChannels,
        _nInputHeight,
        _nInputWidth,
        _nKernelSize,
        _nStride,
        _nOutputHeight,
        _nOutputWidth
    );
    //
    return { spmOutput };
}

void
Pooling::backward(
    const std::vector<const cuMat*>& c_lpmOutputGrads,
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs,
    const std::vector<std::shared_ptr<Tensor>>& c_spmOutputs
)
{
    (void)c_spmOutputs;
    //
    if( (c_spmInputs.size()!=1)||!c_spmInputs[0] )
    {
        throw std::runtime_error(
            "Pooling::backward: exactly one non-null input is required"
        );
    }
    const cuMat& c_mOutputGrad  =singleGrad( c_lpmOutputGrads, "Pooling::backward" );
    const int nOutputRows       =_nOutputHeight * _nOutputWidth * _nChannels;
    if( (c_mOutputGrad._nRows!=nOutputRows)||
        (c_mOutputGrad._nCols!=c_spmInputs[0]->_mData._nCols) )
    {
        throw std::runtime_error(
            "Pooling::backward: output gradient shape mismatch"
        );
    }
    if( static_cast<long long>(nOutputRows)*c_spmInputs[0]->_mData._nCols>INT_MAX )
    {
        throw std::overflow_error(
            "Pooling::backward: gradient exceeds int range"
        );
    }
    // a(n,y_out,x_out,c) = argmax_(k_y,k_x) X[n,y_out*S+k_y,x_out*S+k_x,c]
    // dL/dX[n,y,x,c] += sum_(y_out,x_out) 1[(y,x)=a] * dL/dY[n,y_out,x_out,c]
    // 最大値が同値なら(k_y,k_x)の走査順で最初の位置をaとする。
    cuda_MaxPool2D_backward(
        c_spmInputs[0]->_mGrad,
        c_spmInputs[0]->_mData,
        c_mOutputGrad,
        _nChannels,
        _nInputHeight,
        _nInputWidth,
        _nKernelSize,
        _nStride,
        _nOutputHeight,
        _nOutputWidth
    );
}
