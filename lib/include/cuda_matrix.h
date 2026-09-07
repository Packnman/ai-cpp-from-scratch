#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <concepts>
#include <stdexcept>
#include <vector>

class Mat;
namespace cu_detail {
void fillOnes(float* destination,std::size_t elements);
}

template<class T>
concept cuElement =std::same_as<T,float>||std::same_as<T,std::int32_t>;

// --------------------------
// cuStorage
// --------------------------
template<cuElement T>
class cuStorage
{
public:
    explicit cuStorage(std::size_t elements =0);
    ~cuStorage();

    cuStorage(const cuStorage&) =delete;
    cuStorage& operator=(const cuStorage&) =delete;

private:
    T* _device;
    std::size_t _elements;
public:
    T* data() const noexcept { return _device; }
    std::size_t size() const noexcept { return _elements; }
};

// --------------------------
// cuMat
// --------------------------
template<cuElement T>
class cuMat
{
public:
    cuMat();
    cuMat(int rows,int cols);
    explicit cuMat(const std::vector<std::int64_t>& shape);
    cuMat(std::initializer_list<std::int64_t> shape);
    cuMat(const cuMat& value);

    cuMat(cuMat&& value) noexcept =default;
    ~cuMat() =default;
    cuMat& operator=(const cuMat& value);
    cuMat& operator=(cuMat&& value) noexcept =default;

private:
    std::shared_ptr<cuStorage<T>> _storage;
    std::vector<std::int64_t> _shape;
    std::vector<std::int64_t> _strides;
    std::size_t _offset;
    cuMat(std::shared_ptr<cuStorage<T>> storage,
               std::vector<std::int64_t> shape,
               std::vector<std::int64_t> strides,std::size_t offset);

public:
    const std::vector<std::int64_t>& shape() const noexcept { return _shape; }
    const std::vector<std::int64_t>& strides() const noexcept { return _strides; }
    std::size_t dim() const noexcept { return _shape.size(); }
    std::int64_t size(std::size_t dimension) const;
    std::size_t numel() const noexcept;
    std::size_t offset() const noexcept { return _offset; }
    bool isContiguous() const noexcept;
    int rows() const;
    int cols() const;
    T* data() noexcept;
    const T* data() const noexcept;

public:
    cuMat reshape(const std::vector<std::int64_t>& shape) const;
    cuMat reshape(std::initializer_list<std::int64_t> shape) const;
    cuMat permute(const std::vector<std::size_t>& dimensions) const;
    cuMat slice(std::size_t dimension,std::int64_t start,
                     std::int64_t end,std::int64_t step =1) const;
    cuMat contiguous() const;

    void copyToHost(T* destination,std::size_t elements) const;
    void copyFromHost(const T* source,std::size_t elements);
    std::vector<T> toHost() const;
    void upload(Mat& host) const requires std::same_as<T,float>;
    void download(const Mat& host) requires std::same_as<T,float>;
    void ones() requires std::same_as<T,float>
    {
        if( !isContiguous() )
            throw std::invalid_argument("cuMat::ones: contiguous tensor required");
        cu_detail::fillOnes(data(),numel());
    }

};

// --------------------------
// Type aliases
// --------------------------
using cufStorage =cuStorage<float>;
using cunStorage =cuStorage<std::int32_t>;
using cufMat =cuMat<float>;
using cunMat =cuMat<std::int32_t>;

// --------------------------
// CUDA matrix operations
// --------------------------
void cuda_axpy(cufMat& result,float alpha,const cufMat& a);
void cuda_geam(cufMat& result,float alpha,const cufMat& a,
               float beta,const cufMat& b);
void cuda_gemm(cufMat& result,const cufMat& a,const cufMat& b,
               bool transposeA=false,bool transposeB=false,
               float alpha=1.0f,float beta=0.0f);
void cuda_fill(cufMat& mResult,float fValue);
void cuda_transpose(cufMat& result,const cufMat& a);
void cuda_scale(cufMat& result,float value);
void cuda_mul_elementwise(cufMat& mResult,const cufMat& c_mA,
                          const cufMat& c_mB);
void cuda_ReLU_forward(cufMat& mResult,const cufMat& c_mValue);
void cuda_ReLU_backward(cufMat& mResult,const cufMat& c_mData,
                        const cufMat& c_mGrad);
void cuda_GELU_forward(cufMat& mResult,const cufMat& c_mValue);
void cuda_GELU_backward(cufMat& mResult,const cufMat& c_mData,
                        const cufMat& c_mGrad);
void cuda_Dropout_forward(cufMat& mResult,const cufMat& c_mValue,
                          cufMat& mMask,float fDropProbability);
void cuda_Dropout_backward(cufMat& mResult,const cufMat& c_mGrad,
                           const cufMat& c_mMask);
void cuda_BatchNorm_forward_training(
    cufMat& mResult,const cufMat& c_mValue,
    const cufMat& c_mGamma,const cufMat& c_mBeta,
    cufMat& mRunningMean,cufMat& mRunningVar,
    cufMat& mNormalized,cufMat& mInvStd,
    float fMomentum,float fEpsilon
);
void cuda_BatchNorm_forward_evaluation(
    cufMat& mResult,const cufMat& c_mValue,
    const cufMat& c_mGamma,const cufMat& c_mBeta,
    const cufMat& c_mRunningMean,const cufMat& c_mRunningVar,
    cufMat& mNormalized,cufMat& mInvStd,float fEpsilon
);
void cuda_BatchNorm_backward(
    cufMat& mInputGrad,cufMat& mGammaGrad,cufMat& mBetaGrad,
    const cufMat& c_mOutputGrad,const cufMat& c_mGamma,
    const cufMat& c_mNormalized,const cufMat& c_mInvStd,
    bool isTraining
);
void cuda_SoftmaxCrossEntropy_forward(
    cufMat& mResult,const cufMat& c_mLogits,
    const cufMat& c_mTarget
);
void cuda_SoftmaxCrossEntropy_backward(
    cufMat& mLogitsGrad,const cufMat& c_mLogits,
    const cufMat& c_mTarget,const cufMat& c_mGrad
);
void cuda_Conv2D_im2col(
    cufMat& mResult,const cufMat& c_mInput,
    int nInputChannels,int nInputHeight,int nInputWidth,int nKernelSize,
    int nStride,int nPadding,int nOutputHeight,int nOutputWidth
);
void cuda_Conv2D_pack_output(
    cufMat& mResult,const cufMat& c_mGemm,
    const cufMat& c_mBias,int nOutputHeight,int nOutputWidth
);
void cuda_Conv2D_unpack_grad(
    cufMat& mResult,const cufMat& c_mOutputGrad,
    int nOutputChannels,int nOutputHeight,int nOutputWidth
);
void cuda_Conv2D_col2im(
    cufMat& mResult,const cufMat& c_mColumnGrad,
    int nInputChannels,int nInputHeight,int nInputWidth,int nKernelSize,
    int nStride,int nPadding,int nOutputHeight,int nOutputWidth
);
void cuda_Conv2D_bias_backward(
    cufMat& mResult,const cufMat& c_mGemmGrad
);
void cuda_Pooling_forward(
    cufMat& mResult,const cufMat& c_mInput,
    int nChannels,int nInputHeight,int nInputWidth,int nKernelSize,
    int nStride,int nOutputHeight,int nOutputWidth
);
void cuda_Pooling_backward(
    cufMat& mResult,const cufMat& c_mInput,
    const cufMat& c_mOutputGrad,int nChannels,int nInputHeight,
    int nInputWidth,int nKernelSize,int nStride,
    int nOutputHeight,int nOutputWidth
);
void cuda_Adam_update(
    cufMat& mData,const cufMat& c_mGrad,
    cufMat& mFirstMoment,cufMat& mSecondMoment,
    float fLearningRate,float fBeta1,float fBeta2,
    float fBeta1Correction,float fBeta2Correction,float fEpsilon
);
