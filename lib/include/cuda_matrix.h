#pragma once

#define IDX2F(nRow,nCol,nRows)   ((nCol)*(nRows)+(nRow))      // column-major(Fortran)
#define IDX2C(nRow,nCol,nCols)   ((nRow)*(nCols)+(nCol))      // row-major(C/C++)

class Mat;

class cuMat{
public:
    cuMat();
    cuMat(int nRows,int nCols);
    cuMat(const cuMat& c_mValue);
    cuMat(cuMat&& mValue) noexcept;
    ~cuMat();
private:
public:
    float*  _lpfDevice;
    int     _nRows;
    int     _nCols;
public:
    void upload(Mat& mHost) const;
    void download(const Mat& c_mHost);
    void ones();
    
public:
    cuMat& operator=(const cuMat& c_mValue);         // コピー演算子
    cuMat& operator=(cuMat&& mValue) noexcept;       // ムーブ演算子
};

void cuda_axpy(cuMat& mResult,float fAlpha,const cuMat& c_mA);  // R += αA
void cuda_geam(cuMat& mResult,float fAlpha,const cuMat& c_mA,float fBeta,const cuMat& c_mB);  // R = αA + βB
// R = α*op(A)*op(B) + β*op(B)
void cuda_gemm(cuMat& mResult,const cuMat& c_mA,const cuMat& c_mB,
               bool isTransposeA=false,
               bool isTransposeB=false,
               float fAlpha=1.0f,
               float fBeta=0.0f
);
void cuda_fill(cuMat& mResult,float fValue); // R[:] = value
void cuda_transpose(cuMat& mResult,const cuMat& c_mA); // R = A^T
void cuda_scale(cuMat& mResult,float fValue);    // R *= value
void cuda_mul_elementwise(cuMat& mResult,const cuMat& c_mA,const cuMat& c_mB);  // R = A ⦿ B



void cuda_ReLU_forward(cuMat& mResult,const cuMat& c_mValue);
void cuda_ReLU_backward(
    cuMat& mResult,         // 求める勾配 ∂L/∂x を加算する先
    const cuMat& c_mData,   // ReLUへの入力 x
    const cuMat& c_mGrad    // 上流から来た勾配 ∂L/∂y
);
void cuda_GELU_forward(cuMat& mResult,const cuMat& c_mValue);
void cuda_GELU_backward(
    cuMat& mResult,
    const cuMat& c_mData,
    const cuMat& c_mGrad
);
void cuda_Dropout_forward(
    cuMat& mResult,
    const cuMat& c_mValue,
    cuMat& mMask,
    float fDropProbability
);
void cuda_Dropout_backward(
    cuMat& mResult,
    const cuMat& c_mGrad,
    const cuMat& c_mMask
);
void cuda_BatchNorm_forward_training(
    cuMat& mResult,
    const cuMat& c_mValue,
    const cuMat& c_mGamma,
    const cuMat& c_mBeta,
    cuMat& mRunningMean,
    cuMat& mRunningVar,
    cuMat& mNormalized,
    cuMat& mInvStd,
    float fMomentum,
    float fEpsilon
);
void cuda_BatchNorm_forward_evaluation(
    cuMat& mResult,
    const cuMat& c_mValue,
    const cuMat& c_mGamma,
    const cuMat& c_mBeta,
    const cuMat& c_mRunningMean,
    const cuMat& c_mRunningVar,
    cuMat& mNormalized,
    cuMat& mInvStd,
    float fEpsilon
);
void cuda_BatchNorm_backward(
    cuMat& mInputGrad,
    cuMat& mGammaGrad,
    cuMat& mBetaGrad,
    const cuMat& c_mOutputGrad,
    const cuMat& c_mGamma,
    const cuMat& c_mNormalized,
    const cuMat& c_mInvStd,
    bool isTraining
);
void cuda_SoftmaxCrossEntropy_forward(
    cuMat& mResult,
    const cuMat& c_mLogits,
    const cuMat& c_mTarget
);
void cuda_SoftmaxCrossEntropy_backward(
    cuMat& mLogitsGrad,
    const cuMat& c_mLogits,
    const cuMat& c_mTarget,
    const cuMat& c_mGrad
);

// Spatial operators use HWC-flattened image rows and batch columns.
void cuda_Conv2D_im2col(
    cuMat& mColumns,
    const cuMat& c_mInput,
    int nInputChannels,
    int nInputHeight,
    int nInputWidth,
    int nKernelSize,
    int nStride,
    int nPadding,
    int nOutputHeight,
    int nOutputWidth
);
void cuda_Conv2D_pack_output(
    cuMat& mOutput,
    const cuMat& c_mGemm,
    const cuMat& c_mBias,
    int nOutputHeight,
    int nOutputWidth
);
void cuda_Conv2D_unpack_grad(
    cuMat& mGemmGrad,
    const cuMat& c_mOutputGrad,
    int nOutputChannels,
    int nOutputHeight,
    int nOutputWidth
);
void cuda_Conv2D_col2im(
    cuMat& mInputGrad,
    const cuMat& c_mColumnGrad,
    int nInputChannels,
    int nInputHeight,
    int nInputWidth,
    int nKernelSize,
    int nStride,
    int nPadding,
    int nOutputHeight,
    int nOutputWidth
);
void cuda_Conv2D_bias_backward(
    cuMat& mBiasGrad,
    const cuMat& c_mGemmGrad
);
void cuda_MaxPool2D_forward(
    cuMat& mOutput,
    const cuMat& c_mInput,
    int nChannels,
    int nInputHeight,
    int nInputWidth,
    int nKernelSize,
    int nStride,
    int nOutputHeight,
    int nOutputWidth
);
void cuda_MaxPool2D_backward(
    cuMat& mInputGrad,
    const cuMat& c_mInput,
    const cuMat& c_mOutputGrad,
    int nChannels,
    int nInputHeight,
    int nInputWidth,
    int nKernelSize,
    int nStride,
    int nOutputHeight,
    int nOutputWidth
);

void cuda_Adam_update(
    cuMat& mData,
    const cuMat& c_mGrad,
    cuMat& mFirstMoment,
    cuMat& mSecondMoment,
    float fLearningRate,
    float fBeta1,
    float fBeta2,
    float fBeta1Correction,
    float fBeta2Correction,
    float fEpsilon
);
