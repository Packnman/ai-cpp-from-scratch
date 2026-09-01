#include "cuda_function.h"
#include "cuda_tensor.h"

#include <climits>
#include <stdexcept>
#include <string>

namespace
{
const cuMat& singleGrad(
    const TensorGradList& c_lpmOutputGrads,
    const char* c_lpszFunctionName
)
{
    if( (c_lpmOutputGrads.size()!=1)||(c_lpmOutputGrads[0]==nullptr) )
    {
        throw std::runtime_error(
            std::string( c_lpszFunctionName ) +
            ": exactly one output gradient is required"
        );
    }
    return *c_lpmOutputGrads[0];
}
} // namespace

Conv2D::Conv2D( Tensor* lpWeight, Tensor* lpBias, int nInputChannels, int nInputHeight,
                int nInputWidth, int nKernelSize, int nStride, int nPadding )
    :_lpmWeight( lpWeight ),
     _lpmBias( lpBias ),
     _nInputChannels( nInputChannels ),
     _nInputHeight( nInputHeight ),
     _nInputWidth( nInputWidth ),
     _nKernelSize( nKernelSize ),
     _nStride( nStride ),
     _nPadding( nPadding ),
     _nOutputChannels( 0 ),
     _nOutputHeight( 0 ),
     _nOutputWidth( 0 )
{
    if( !lpWeight||!lpBias )
    {
        throw std::invalid_argument(
            "Conv2D: weight and bias must not be null"
        );
    }
    if( (nInputChannels<=0)||(nInputHeight<=0)||(nInputWidth<=0)||
        (nKernelSize<=0)||(nStride<=0)||(nPadding<0) )
    {
        throw std::invalid_argument(
            "Conv2D: invalid channels, image, kernel, stride, or padding"
        );
    }
    const long long nPaddedHeight   =static_cast<long long>( nInputHeight ) + 2LL * nPadding;
    const long long nPaddedWidth    =static_cast<long long>( nInputWidth ) + 2LL * nPadding;
    if( (nPaddedHeight<nKernelSize)||(nPaddedWidth<nKernelSize) )
    {
        throw std::invalid_argument(
            "Conv2D: kernel exceeds padded input"
        );
    }
    // H_out = floor((H_in + 2P - K) / S) + 1
    // W_out = floor((W_in + 2P - K) / S) + 1
    const long long nOutputHeight   =(nPaddedHeight - nKernelSize) / nStride + 1;
    const long long nOutputWidth    =(nPaddedWidth - nKernelSize) / nStride + 1;
    const long long nPatchSize      =static_cast<long long>( nKernelSize ) * nKernelSize * nInputChannels;
    const long long nInputRows      =static_cast<long long>( nInputHeight ) * nInputWidth * nInputChannels;
    if( (nOutputHeight>INT_MAX)||(nOutputWidth>INT_MAX)||
        (nPatchSize>INT_MAX)||(nInputRows>INT_MAX) )
    {
        throw std::overflow_error(
            "Conv2D: dimensions exceed int range"
        );
    }
    if( (lpWeight->_mData._nRows<=0)||(lpWeight->_mData._nCols!=nPatchSize) )
    {
        throw std::invalid_argument( "Conv2D: invalid weight shape" );
    }
    _nOutputChannels    =lpWeight->_mData._nRows;
    if( (lpBias->_mData._nRows!=_nOutputChannels)||(lpBias->_mData._nCols!=1) )
    {
        throw std::invalid_argument( "Conv2D: invalid bias shape" );
    }
    _nOutputHeight  =static_cast<int>( nOutputHeight );
    _nOutputWidth   =static_cast<int>( nOutputWidth );
}
Conv2D::~Conv2D() = default;

TensorList Conv2D::forward(const TensorList& c_spmInputs)
{
    if( (c_spmInputs.size()!=1)||!c_spmInputs[0] )
    {
        throw std::runtime_error(
            "Conv2D::forward: exactly one non-null input is required"
        );
    }
    const long long nInputRows  =static_cast<long long>( _nInputHeight ) * _nInputWidth * _nInputChannels;
    const long long nPositions  =static_cast<long long>( _nOutputHeight ) * _nOutputWidth;
    const long long nOutputRows =nPositions * _nOutputChannels;
    const long long nCombined   =nPositions * c_spmInputs[0]->_mData._nCols;
    if( (c_spmInputs[0]->_mData._nRows!=nInputRows)||(c_spmInputs[0]->_mData._nCols<=0) )
    {
        throw std::runtime_error( "Conv2D::forward: input shape mismatch" );
    }
    if( (nOutputRows>INT_MAX)||(nCombined>INT_MAX)||
        (nOutputRows*c_spmInputs[0]->_mData._nCols>INT_MAX)||
        (nCombined*_lpmWeight->_mData._nCols>INT_MAX)||
        (nCombined*_nOutputChannels>INT_MAX) )
    {
        throw std::overflow_error( "Conv2D::forward: size exceeds int range" );
    }
    cuMat mColumns( _lpmWeight->_mData._nCols, static_cast<int>( nCombined ) );
    cuMat mProduct( _nOutputChannels, static_cast<int>( nCombined ) );
    //
    // Y[n,y_out,x_out,c_o] = b[c_o]
    //   + sum_(k_y,k_x,c_i) W[c_o,(k_y*K+k_x)*C_in+c_i]
    //                           * X[n,y_out*S+k_y-P,x_out*S+k_x-P,c_i]
    // 出力位置 o = y_out * W_out + x_out ごとの局所領域を列へ展開する。
    // C[(k_y*K+k_x)*C_in+c_i, n*(H_out*W_out)+o]
    //   = X[n, y_out*S+k_y-P, x_out*S+k_x-P, c_i]
    // 範囲外のXは0とする。
    cuda_Conv2D_im2col(
        mColumns,
        c_spmInputs[0]->_mData,
        _nInputChannels,
        _nInputHeight,
        _nInputWidth,
        _nKernelSize,
        _nStride,
        _nPadding,
        _nOutputHeight,
        _nOutputWidth
    );
    // P = W * C
    // Y[n,o,c_o] = P[c_o,n*(H_out*W_out)+o] + b[c_o]
    cuda_gemm( mProduct, _lpmWeight->_mData, mColumns );
    auto spmOutput  =std::make_shared<Tensor>( static_cast<int>(nOutputRows), c_spmInputs[0]->_mData._nCols );
    cuda_Conv2D_pack_output(
        spmOutput->_mData,
        mProduct,
        _lpmBias->_mData,
        _nOutputHeight,
        _nOutputWidth
    );
    //
    return { spmOutput };
}

void Conv2D::backward(
    const TensorGradList& c_lpmOutputGrads,
    const TensorList& c_spmInputs,
    const TensorList& c_spmOutputs
)
{
    (void)c_spmOutputs;
    //
    if( (c_spmInputs.size()!=1)||!c_spmInputs[0] )
    {
        throw std::runtime_error(
            "Conv2D::backward: exactly one non-null input is required"
        );
    }
    const cuMat& c_mOutputGrad  =singleGrad( c_lpmOutputGrads, "Conv2D::backward" );
    const long long nPositions  =static_cast<long long>( _nOutputHeight ) * _nOutputWidth;
    const long long nCombined64 =nPositions * c_spmInputs[0]->_mData._nCols;
    if( (nCombined64>INT_MAX)||(nCombined64*_lpmWeight->_mData._nCols>INT_MAX)||
        (nCombined64*_nOutputChannels>INT_MAX) )
    {
        throw std::overflow_error( "Conv2D::backward: workspace exceeds int range" );
    }
    const int nCombined =static_cast<int>( nCombined64 );

    if( (c_mOutputGrad._nRows!=nPositions*_nOutputChannels)||
        (c_mOutputGrad._nCols!=c_spmInputs[0]->_mData._nCols) )
    {
        throw std::runtime_error(
            "Conv2D::backward: output gradient shape mismatch"
        );
    }
    cuMat mColumns( _lpmWeight->_mData._nCols, nCombined );
    cuMat mProductGrad( _nOutputChannels, nCombined );
    cuMat mColumnGrad( _lpmWeight->_mData._nCols, nCombined );

    cuda_Conv2D_im2col(
        mColumns,
        c_spmInputs[0]->_mData,
        _nInputChannels,
        _nInputHeight,
        _nInputWidth,
        _nKernelSize,
        _nStride,
        _nPadding,
        _nOutputHeight,
        _nOutputWidth
    );
    // G[c_o,n*(H_out*W_out)+o] = dL/dY[n,o,c_o]
    cuda_Conv2D_unpack_grad(
        mProductGrad,
        c_mOutputGrad,
        _nOutputChannels,
        _nOutputHeight,
        _nOutputWidth
    );
    // dL/dW += G * C^T
    cuda_gemm(
        _lpmWeight->_mGrad,
        mProductGrad,
        mColumns,
        false,
        true,
        1.0f,
        1.0f
    );
    // dL/dC = W^T * G
    cuda_gemm(
        mColumnGrad,
        _lpmWeight->_mData,
        mProductGrad,
        true,
        false
    );
    // dL/dX += col2im(dL/dC)
    // 同じ入力要素を参照する全kernel位置・出力位置の寄与を加算する。
    cuda_Conv2D_col2im(
        c_spmInputs[0]->_mGrad,
        mColumnGrad, 
        _nInputChannels,
        _nInputHeight,
        _nInputWidth,
        _nKernelSize,
        _nStride,
        _nPadding,
        _nOutputHeight,
        _nOutputWidth
    );
    // dL/db[c_o] += sum_(n,o) G[c_o,n*(H_out*W_out)+o]
    cuda_Conv2D_bias_backward( _lpmBias->_mGrad,mProductGrad );
}

MaxPool2D::MaxPool2D( int nChannels, int nInputHeight, int nInputWidth, int nKernelSize,
                      int nStride )
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
            "MaxPool2D: invalid channels, image, kernel, or stride"
        );
    }
    if( (nKernelSize>nInputHeight)||(nKernelSize>nInputWidth) )
    {
        throw std::invalid_argument( "MaxPool2D: kernel exceeds input" );
    }
    const long long nInputRows      =static_cast<long long>( nInputHeight ) * nInputWidth * nChannels;
    // H_out = floor((H_in - K) / S) + 1
    // W_out = floor((W_in - K) / S) + 1
    const long long nOutputHeight   =( nInputHeight - nKernelSize ) / nStride + 1;
    const long long nOutputWidth    =( nInputWidth - nKernelSize ) / nStride + 1;
    if( (nInputRows>INT_MAX)||(nOutputHeight*nOutputWidth*nChannels>INT_MAX) )
    {
        throw std::overflow_error(
            "MaxPool2D: dimensions exceed int range"
        );
    }
    _nOutputHeight  =static_cast<int>( nOutputHeight );
    _nOutputWidth   =static_cast<int>( nOutputWidth );
}
MaxPool2D::~MaxPool2D() = default;

TensorList MaxPool2D::forward( const TensorList& c_spmInputs )
{
    if( (c_spmInputs.size()!=1)||!c_spmInputs[0] )
    {
        throw std::runtime_error(
            "MaxPool2D::forward: exactly one non-null input is required"
        );
    }
    const int nInputRows    =_nInputHeight * _nInputWidth * _nChannels;
    const int nOutputRows   =_nOutputHeight * _nOutputWidth * _nChannels;
    if( (c_spmInputs[0]->_mData._nRows!=nInputRows)||(c_spmInputs[0]->_mData._nCols<=0) )
    {
        throw std::runtime_error( "MaxPool2D::forward: input shape mismatch" );
    }
    if( static_cast<long long>(nOutputRows)*c_spmInputs[0]->_mData._nCols>INT_MAX )
    {
        throw std::overflow_error(
            "MaxPool2D::forward: output exceeds int range"
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

void MaxPool2D::backward( const TensorGradList& c_lpmOutputGrads, const TensorList& c_spmInputs,
                          const TensorList& c_spmOutputs )
{
    (void)c_spmOutputs;
    //
    if( (c_spmInputs.size()!=1)||!c_spmInputs[0] )
    {
        throw std::runtime_error(
            "MaxPool2D::backward: exactly one non-null input is required"
        );
    }
    const cuMat& c_mOutputGrad  =singleGrad( c_lpmOutputGrads, "MaxPool2D::backward" );
    const int nOutputRows       =_nOutputHeight * _nOutputWidth * _nChannels;
    if( (c_mOutputGrad._nRows!=nOutputRows)||
        (c_mOutputGrad._nCols!=c_spmInputs[0]->_mData._nCols) )
    {
        throw std::runtime_error(
            "MaxPool2D::backward: output gradient shape mismatch"
        );
    }
    if( static_cast<long long>(nOutputRows)*c_spmInputs[0]->_mData._nCols>INT_MAX )
    {
        throw std::overflow_error(
            "MaxPool2D::backward: gradient exceeds int range"
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
