#include "cuda_matrix.h"
#include "cuda_bublas.h"
#include "matrix.h"

#include <algorithm>
#include <climits>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
void checkCuda(cudaError_t status,const char* operation)
{
    if(status!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}
std::size_t checkedNumel(const std::vector<std::int64_t>& shape)
{
    if(shape.empty()) return 0;
    std::size_t result=1;
    for(std::int64_t extent:shape) {
        if(extent<0) throw std::invalid_argument("cuMat: negative shape extent");
        if(extent==0) return 0;
        if(static_cast<std::uint64_t>(extent)>SIZE_MAX/result) throw std::overflow_error("cuMat: shape is too large");
        result*=static_cast<std::size_t>(extent);
    }
    return result;
}
std::vector<std::int64_t> rowMajorStrides(const std::vector<std::int64_t>& shape)
{
    std::vector<std::int64_t> strides(shape.size(),1);
    for(std::size_t i=shape.size();i>1;--i) {
        if(shape[i-1] && strides[i-1]>INT64_MAX/shape[i-1]) throw std::overflow_error("cuMat: stride is too large");
        strides[i-2]=strides[i-1]*shape[i-1];
    }
    return strides;
}
template<cuElement T>
void requireContiguous(const cuMat<T>& value,const char* operation)
{
    if(!value.isContiguous()) throw std::invalid_argument(std::string(operation)+": contiguous tensor required");
}
void requireSameShape(const cufMat& a,const cufMat& b,const char* operation)
{
    if(a.shape()!=b.shape()) throw std::runtime_error(std::string(operation)+": shape mismatch");
}
} // anonymous namespace




template<cuElement T>
cuStorage<T>::cuStorage(std::size_t elements)
    :_device(nullptr),_elements(elements)
{
    if( elements>std::numeric_limits<std::size_t>::max()/sizeof(T) )
        throw std::overflow_error("cuStorage: allocation is too large");
    if( elements )
    {
        checkCuda(
            cudaMalloc(reinterpret_cast<void**>(&_device),elements*sizeof(T)),
            "cuStorage cudaMalloc"
        );
    }
}
template<cuElement T>
cuStorage<T>::~cuStorage()
{
    if(_device) cudaFree(_device);
}

template<cuElement T>
cuMat<T>::cuMat()
    :_offset(0)
{}
template<cuElement T>
cuMat<T>::cuMat(int rows,int cols)
    :cuMat(std::vector<std::int64_t>{rows,cols})
{}
template<cuElement T>
cuMat<T>::cuMat(std::initializer_list<std::int64_t> shape)
    :cuMat(std::vector<std::int64_t>(shape))
{}
template<cuElement T>
cuMat<T>::cuMat(const std::vector<std::int64_t>& shape)
    :_storage(std::make_shared<cuStorage<T>>(checkedNumel(shape))),
     _shape(shape),
     _strides(rowMajorStrides(shape)),
     _offset(0)
{}
template<cuElement T>
cuMat<T>::cuMat(std::shared_ptr<cuStorage<T>> storage,
                std::vector<std::int64_t> shape,
                std::vector<std::int64_t> strides,std::size_t offset)
    :_storage(std::move(storage)),
     _shape(std::move(shape)),
     _strides(std::move(strides)),
     _offset(offset)
{}
template<cuElement T>
cuMat<T>::cuMat(const cuMat& value)
    :cuMat(value.shape())
{
    auto host   =value.toHost();
    copyFromHost(host.data(),host.size());
}
template<cuElement T>
cuMat<T>& cuMat<T>::operator=(const cuMat& value)
{
    if( this!=&value )
    {
        cuMat copy(value);
        *this=std::move(copy);
    }
    return *this;
}
template<cuElement T>
std::int64_t cuMat<T>::size(std::size_t dimension) const
{
    if( dimension>=dim() )
    {
        throw std::out_of_range("cuMat::size: dimension out of range");
    }
    return _shape[dimension];
}
template<cuElement T>
std::size_t cuMat<T>::numel() const noexcept
{
    if( _shape.empty() )
    {
        return 0;
    }
    std::size_t result=1;
    for( auto extent:_shape )
    {
        result*=static_cast<std::size_t>(extent);
    }
    return result;
}
template<cuElement T>
bool cuMat<T>::isContiguous() const noexcept
{
    if( numel()==0 ) return true;
    std::int64_t expected=1;
    for( std::size_t i=_shape.size();i>0;--i )
    {
        if( (_shape[i-1]>1)&&(_strides[i-1]!=expected) )
        {
            return false;
        }
        if( i>1&&_shape[i-1] )
        {
            if( expected>INT64_MAX/_shape[i-1] ) return false;
            expected*=_shape[i-1];
        }
    }
    return true;
}
template<cuElement T>
int cuMat<T>::rows() const
{
    if( (dim()!=2)||(size(0)>INT_MAX) )
    {
        throw std::runtime_error("cuMat::rows: rank-2 tensor required");
    }
    return static_cast<int>(size(0));
}
template<cuElement T>
int cuMat<T>::cols() const
{
    if( (dim()!=2)||(size(1)>INT_MAX) )
    {
        throw std::runtime_error("cuMat::cols: rank-2 tensor required");
    }
    return static_cast<int>(size(1));
}
template<cuElement T>
T* cuMat<T>::data() noexcept
{
    return _storage&&_storage->data() ? _storage->data()+_offset : nullptr;
}
template<cuElement T>
const T* cuMat<T>::data() const noexcept
{
    return _storage&&_storage->data() ? _storage->data()+_offset : nullptr;
}
template<cuElement T>
cuMat<T> cuMat<T>::reshape(const std::vector<std::int64_t>& shape) const
{
    if( !isContiguous() )
    {
        throw std::invalid_argument("cuMat::reshape: contiguous tensor required");
    }
    if( checkedNumel(shape)!=numel() )
    {
        throw std::invalid_argument("cuMat::reshape: element count mismatch");
    }
    return cuMat(_storage,shape,rowMajorStrides(shape),_offset);
}
template<cuElement T>
cuMat<T> cuMat<T>::reshape(std::initializer_list<std::int64_t> shape) const
{
    return reshape(std::vector<std::int64_t>(shape));
}
template<cuElement T>
cuMat<T> cuMat<T>::permute(const std::vector<std::size_t>& dimensions) const
{
    if( dimensions.size()!=dim() )
    {
        throw std::invalid_argument("cuMat::permute: rank mismatch");
    }
    std::vector<bool> used(dim());
    std::vector<std::int64_t> shape(dim());
    std::vector<std::int64_t> strides(dim());
    for( std::size_t i=0;i<dim();++i )
    {
        if( (dimensions[i]>=dim())||(used[dimensions[i]]) )
        {
            throw std::invalid_argument("cuMat::permute: invalid permutation");
        }
        used[dimensions[i]] =true;
        shape[i]            =_shape[dimensions[i]];
        strides[i]          =_strides[dimensions[i]];
    }
    return cuMat(_storage,std::move(shape),std::move(strides),_offset);
}
template<cuElement T>
cuMat<T> cuMat<T>::slice(
    std::size_t dimension,
    std::int64_t start,
    std::int64_t end,
    std::int64_t step
) const
{
    if( dimension>=dim() )
    {
        throw std::out_of_range("cuMat::slice: dimension out of range");
    }
    if( (step<=0)||(start<0)||(end<start)||(end>_shape[dimension]) )
    {
        throw std::invalid_argument("cuMat::slice: invalid range");
    }
    if( _strides[dimension]>INT64_MAX/step )
        throw std::overflow_error("cuMat::slice: stride is too large");
    auto shape =_shape;
    auto strides =_strides;
    shape[dimension] =(end-start)/step+((end-start)%step!=0);
    const auto delta =static_cast<std::size_t>(start)*
                      static_cast<std::size_t>(_strides[dimension]);
    if( delta>std::numeric_limits<std::size_t>::max()-_offset )
        throw std::overflow_error("cuMat::slice: offset is too large");
    const std::size_t offset =_offset+delta;
    strides[dimension] *=step;

    return cuMat(_storage,std::move(shape),std::move(strides),offset);
}
template<cuElement T>
std::vector<T> cuMat<T>::toHost() const
{
    std::vector<T> result(numel());
    if(!numel()) return result;
    if(isContiguous())
    {
        checkCuda(
            cudaMemcpy(
                result.data(),
                data(),
                numel()*sizeof(T),
                cudaMemcpyDeviceToHost
            ),
            "cuMat::toHost"
        );
        return result;
    }
    std::vector<T> storage(_storage->size());
    checkCuda(
        cudaMemcpy(
            storage.data(),
            _storage->data(),
            storage.size()*sizeof(T),
            cudaMemcpyDeviceToHost
        ),
        "cuMat::toHost view"
    );
    for( std::size_t linear=0;linear<numel();++linear )
    {
        std::size_t remainder   =linear;
        std::size_t source      =_offset;
        for( std::size_t d=dim();d>0;--d )
        {
            auto coordinate =remainder%static_cast<std::size_t>(_shape[d-1]);
            remainder       /=static_cast<std::size_t>(_shape[d-1]);
            source          +=coordinate*static_cast<std::size_t>(_strides[d-1]);
        }
        result[linear]  =storage[source];
    }
    return result;
}
template<cuElement T>
cuMat<T> cuMat<T>::contiguous() const
{
    if( isContiguous() )
    {
        return cuMat(_storage,_shape,_strides,_offset);
    }
    cuMat result(_shape);
    auto host   =toHost();
    result.copyFromHost(host.data(),host.size());
    
    return result;
}
template<cuElement T>
void cuMat<T>::copyToHost(T* destination,std::size_t elements) const
{
    if( elements!=numel() )
    {
        throw std::invalid_argument("cuMat::copyToHost: size mismatch");
    }
    auto host =toHost();
    std::copy(host.begin(),host.end(),destination);
}
template<cuElement T>
void cuMat<T>::copyFromHost(const T* source,std::size_t elements)
{
    requireContiguous(*this,"cuMat::copyFromHost");
    if( elements!=numel() )
    {
        throw std::invalid_argument("cuMat::copyFromHost: size mismatch");
    }
    if( elements )
    {
        checkCuda(
            cudaMemcpy(
                data(),
                source,
                elements*sizeof(T),
                cudaMemcpyHostToDevice
            ),
            "cuMat::copyFromHost"
        );
    }
}
template<cuElement T>
void cuMat<T>::upload(Mat& host) const requires std::same_as<T,float>
{
    if( (rows()!=host._nRows)||(cols()!=host._nCols) )
    {
        throw std::invalid_argument("cuMat::upload: shape mismatch");
    }
    copyToHost(host._lpfHost,numel());
}
template<cuElement T>
void cuMat<T>::download(const Mat& host) requires std::same_as<T,float>
{
    if( (rows()!=host._nRows)||(cols()!=host._nCols) )
    {
        throw std::invalid_argument("cuMat::download: shape mismatch");
    }
    copyFromHost(host._lpfHost,numel());
}

template class cuStorage<float>;
template class cuStorage<std::int32_t>;
template class cuMat<float>;
template class cuMat<std::int32_t>;

void cuda_scale(cufMat& result,float value)
{
    requireContiguous(result,"cuda_scale");
    auto status =cublasSscal(
        getCublasHandle(),
        static_cast<int>(result.numel()),
        &value,
        result.data(),
        1
    );
    // Check for errors
    if( status!=CUBLAS_STATUS_SUCCESS )
    {
        throw std::runtime_error("cuda_scale: cublasSscal failed");
    }
}
void cuda_axpy(cufMat& result,float alpha,const cufMat& a)
{
    requireContiguous(result,"cuda_axpy");
    requireContiguous(a,"cuda_axpy");
    requireSameShape(result,a,"cuda_axpy");
    auto status =cublasSaxpy(
        getCublasHandle(),
        static_cast<int>(result.numel()),
        &alpha,
        a.data(),
        1,
        result.data(),
        1
    );
    // Check for errors
    if( status!=CUBLAS_STATUS_SUCCESS )
    {
        throw std::runtime_error("cuda_axpy: cublasSaxpy failed");
    }
}
void cuda_geam(cufMat& result,float alpha,const cufMat& a,float beta,const cufMat& b)
{
    requireContiguous(result,"cuda_geam");
    requireContiguous(a,"cuda_geam");
    requireContiguous(b,"cuda_geam");
    requireSameShape(a,b,"cuda_geam");
    requireSameShape(result,a,"cuda_geam");
    if( result.data()!=a.data() )
    {
        checkCuda(
            cudaMemcpy(
                result.data(),
                a.data(),
                a.numel()*sizeof(float),
                cudaMemcpyDeviceToDevice
            ),
            "cuda_geam copy"
        );
    }
    cuda_scale(result,alpha);
    cuda_axpy(result,beta,b);
}
void cuda_gemm(cufMat& result,const cufMat& a,const cufMat& b,bool transposeA,bool transposeB,float alpha,float beta)
{
    requireContiguous(result,"cuda_gemm");
    requireContiguous(a,"cuda_gemm");
    requireContiguous(b,"cuda_gemm");
    int ar  =transposeA?a.cols():a.rows();
    int ac  =transposeA?a.rows():a.cols();
    int br  =transposeB?b.cols():b.rows();
    int bc  =transposeB?b.rows():b.cols();
    if( (ac!=br)||(result.rows()!=ar)||(result.cols()!=bc) )
    {
        throw std::runtime_error("cuda_gemm: matrix shape mismatch");
    }
    auto status =cublasSgemm(
        getCublasHandle(),
        transposeB?CUBLAS_OP_T:CUBLAS_OP_N,
        transposeA?CUBLAS_OP_T:CUBLAS_OP_N,
        bc,
        ar,
        ac,
        &alpha,
        b.data(),
        b.cols(),
        a.data(),
        a.cols(),
        &beta,
        result.data(),
        result.cols()
    );
    //
    if( status!=CUBLAS_STATUS_SUCCESS )
    {
        throw std::runtime_error("cuda_gemm: cublasSgemm failed");
    }
}
void cuda_transpose(cufMat& result,const cufMat& a)
{
    requireContiguous(result,"cuda_transpose");
    requireContiguous(a,"cuda_transpose");
    if( (result.rows()!=a.cols())||(result.cols()!=a.rows()) )
    {
        throw std::runtime_error("cuda_transpose: matrix shape mismatch");
    }
    float alpha =1;
    float beta  =0;
    auto status =cublasSgeam(
        getCublasHandle(),
        CUBLAS_OP_T,CUBLAS_OP_N,
        a.rows(),
        a.cols(),
        &alpha,
        a.data(),
        a.cols(),
        &beta,
        a.data(),
        a.rows(),
        result.data(),
        result.cols()
    );
    // Check for errors
    if( status!=CUBLAS_STATUS_SUCCESS )
    {
        throw std::runtime_error("cuda_transpose: cublasSgeam failed");
    }
}
