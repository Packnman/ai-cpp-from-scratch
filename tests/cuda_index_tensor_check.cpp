#include "cuda_matrix.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
template<class T>
concept HasFloatOnlyOperations =requires(T value,Mat& host)
{
    value.ones();
    value.upload(host);
    value.download(host);
};

static_assert(cuElement<float>);
static_assert(cuElement<std::int32_t>);
static_assert(!cuElement<double>);
static_assert(std::same_as<cufMat,cuMat<float>>);
static_assert(std::same_as<cunMat,cuMat<std::int32_t>>);
static_assert(std::same_as<cufStorage,cuStorage<float>>);
static_assert(std::same_as<cunStorage,cuStorage<std::int32_t>>);
static_assert(HasFloatOnlyOperations<cufMat>);
static_assert(!HasFloatOnlyOperations<cunMat>);
static_assert(!std::is_convertible_v<cufMat,cunMat>);

void require(bool condition,const char* message)
{
    if( !condition ) throw std::runtime_error(message);
}

template<class Exception,class Function>
void requireThrows(Function&& function,const char* message)
{
    try
    {
        function();
    }
    catch( const Exception& )
    {
        return;
    }
    throw std::runtime_error(message);
}

cunMat survivingView()
{
    cunMat source({2,3,4});
    std::vector<std::int32_t> values(24);
    for( std::int32_t i=0;i<24;++i ) values[i] =i;
    source.copyFromHost(values.data(),values.size());
    return source.slice(1,1,3);
}

void floatDimensionRegression()
{
    cufMat matrix(3,5);
    require(matrix.rows()==3&&matrix.cols()==5,"cufMat rows/cols failed");
    cufMat rankThree({1,2,3});
    requireThrows<std::runtime_error>(
        [&] { (void)rankThree.rows(); },
        "cufMat rows accepted a non-matrix"
    );
    requireThrows<std::runtime_error>(
        [&] { (void)rankThree.cols(); },
        "cufMat cols accepted a non-matrix"
    );
}

void metadataAndTransferChecks()
{
    cunMat tensor({2,3,4});
    require(
        tensor.shape()==std::vector<std::int64_t>({2,3,4}),
        "index shape failed"
    );
    require(
        tensor.strides()==std::vector<std::int64_t>({12,4,1}),
        "index strides failed"
    );
    require(
        tensor.dim()==3&&tensor.size(1)==3&&tensor.numel()==24&&
        tensor.offset()==0&&tensor.isContiguous(),
        "index metadata failed"
    );

    std::vector<std::int32_t> values(24);
    for( std::int32_t i=0;i<24;++i ) values[i] =100+i;
    tensor.copyFromHost(values.data(),values.size());
    require(tensor.toHost()==values,"index host/device round trip failed");

    std::vector<std::int32_t> copied(values.size());
    tensor.copyToHost(copied.data(),copied.size());
    require(copied==values,"index copyToHost failed");

    cunMat matrix(2,3);
    require(matrix.rows()==2&&matrix.cols()==3,"index rows/cols failed");
}

void viewChecks()
{
    cunMat tensor({2,3,4});
    std::vector<std::int32_t> values(24);
    for( std::int32_t i=0;i<24;++i ) values[i] =i;
    tensor.copyFromHost(values.data(),values.size());

    auto reshaped =tensor.reshape({4,6});
    require(
        reshaped.data()==tensor.data()&&
        reshaped.strides()==std::vector<std::int64_t>({6,1}),
        "index reshape view failed"
    );

    auto permuted =tensor.permute({1,0,2});
    const std::vector<std::int32_t> expectedPermuted{
        0,1,2,3,12,13,14,15,4,5,6,7,
        16,17,18,19,8,9,10,11,20,21,22,23
    };
    require(
        !permuted.isContiguous()&&permuted.toHost()==expectedPermuted,
        "index permute logical order failed"
    );
    auto packed =permuted.contiguous();
    require(
        packed.isContiguous()&&packed.toHost()==expectedPermuted,
        "index contiguous failed"
    );

    const std::vector<std::int32_t> expectedSlice{
        4,5,6,7,8,9,10,11,16,17,18,19,20,21,22,23
    };
    auto slice =survivingView();
    require(
        slice.offset()==4&&slice.toHost()==expectedSlice,
        "index slice or storage lifetime failed"
    );

    cunMat deepCopy =permuted;
    std::vector<std::int32_t> replacement(24,-1);
    tensor.copyFromHost(replacement.data(),replacement.size());
    require(deepCopy.toHost()==expectedPermuted,"index copy was not deep");

    cunMat moved =std::move(deepCopy);
    require(moved.toHost()==expectedPermuted,"index move failed");
}

void validationChecks()
{
    requireThrows<std::invalid_argument>(
        [] { cunMat invalid({2,-1}); },
        "negative shape accepted"
    );
    requireThrows<std::overflow_error>(
        [] { cunMat invalid({0,INT64_MAX,2}); },
        "stride overflow accepted"
    );
    requireThrows<std::overflow_error>(
        [] { cunMat invalid({INT64_MAX,3}); },
        "element count overflow accepted"
    );

    cunMat tensor({2,3,4});
    requireThrows<std::out_of_range>(
        [&] { (void)tensor.size(3); },"invalid dimension accepted"
    );
    requireThrows<std::invalid_argument>(
        [&] { (void)tensor.permute({0,0,2}); },
        "duplicate permutation accepted"
    );
    requireThrows<std::invalid_argument>(
        [&] { (void)tensor.slice(1,0,4); },"invalid slice accepted"
    );
    requireThrows<std::invalid_argument>(
        [&] { (void)tensor.reshape({2,2}); },"invalid reshape accepted"
    );
    requireThrows<std::invalid_argument>(
        [&] { tensor.copyFromHost(nullptr,1); },
        "copyFromHost size mismatch accepted"
    );
    requireThrows<std::invalid_argument>(
        [&] { tensor.copyToHost(nullptr,1); },
        "copyToHost size mismatch accepted"
    );
    requireThrows<std::runtime_error>(
        [&] { (void)tensor.rows(); },"rank-3 rows accepted"
    );

    cunMat empty({2,0,4});
    require(
        empty.numel()==0&&empty.data()==nullptr&&empty.toHost().empty(),
        "zero extent handling failed"
    );
    empty.copyFromHost(nullptr,0);
    empty.copyToHost(nullptr,0);
}
} // namespace

int main()
{
    try
    {
        floatDimensionRegression();
        metadataAndTransferChecks();
        viewChecks();
        validationChecks();
        std::cout<<"cuda index tensor check passed\n";
        return 0;
    }
    catch( const std::exception& error )
    {
        std::cerr<<error.what()<<'\n';
        return 1;
    }
}
