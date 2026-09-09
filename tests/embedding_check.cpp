#include "cuda_function_Embedding.h"
#include "cuda_tensor.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition,const char* message)
{
    if( !condition ) throw std::runtime_error(message);
}

template<class Exception,class Callable>
void requireThrows(Callable&& callable,const char* message)
{
    try
    {
        callable();
    }
    catch( const Exception& )
    {
        return;
    }
    throw std::runtime_error(message);
}

std::shared_ptr<cunMat> indices(
    const std::vector<std::int64_t>& shape,
    const std::vector<std::int32_t>& values
)
{
    auto result =std::make_shared<cunMat>( shape );
    result->copyFromHost( values.data(),values.size() );
    return result;
}

void forwardAndBackward()
{
    Tensor weight({2,4});
    const std::vector<float> weightValues{
        10.0f,11.0f,12.0f,13.0f,
        20.0f,21.0f,22.0f,23.0f
    };
    weight._mData.copyFromHost(weightValues.data(),weightValues.size());

    Embedding embedding(&weight);
    auto tokenIds =indices({3,2},{2,1,2,0,3,2});
    auto output =embedding(tokenIds);

    require(
        output->_mData.shape()==std::vector<std::int64_t>({2,3,2}),
        "embedding output shape mismatch"
    );
    const std::vector<float> expected{
        12.0f,11.0f,12.0f,10.0f,13.0f,12.0f,
        22.0f,21.0f,22.0f,20.0f,23.0f,22.0f
    };
    require(output->_mData.toHost()==expected,"embedding gather values mismatch");

    output->backward();
    const std::vector<float> expectedGrad{
        1.0f,1.0f,3.0f,1.0f,
        1.0f,1.0f,3.0f,1.0f
    };
    require(
        weight._mGrad.toHost()==expectedGrad,
        "embedding duplicate-token gradient accumulation mismatch"
    );
}

void validationAndLifetime()
{
    Tensor weight({2,3});
    Embedding embedding(&weight);

    requireThrows<std::out_of_range>(
        [&] { (void)embedding(indices({1,1},{3})); },
        "out-of-range token ID was accepted"
    );
    requireThrows<std::out_of_range>(
        [&] { (void)embedding(indices({1,1},{-1})); },
        "negative token ID was accepted"
    );
    requireThrows<std::invalid_argument>(
        [&] { (void)embedding(indices({1,1,1},{0})); },
        "non-matrix indices were accepted"
    );

    auto tokenIds =indices({2,1},{0,2});
    std::weak_ptr<const cunMat> lifetime =tokenIds;
    auto output =embedding(tokenIds);
    tokenIds.reset();
    require(!lifetime.expired(),"Context did not retain token IDs");
    output->backward();
    output.reset();
    require(lifetime.expired(),"Context retained token IDs after graph release");

    requireThrows<std::invalid_argument>(
        [] { Embedding invalid(nullptr); },
        "null weight was accepted"
    );
    Tensor rankThree({1,2,3});
    requireThrows<std::invalid_argument>(
        [&] { Embedding invalid(&rankThree); },
        "invalid weight rank was accepted"
    );
}
} // namespace

int main()
{
    try
    {
        forwardAndBackward();
        validationAndLifetime();
        std::cout<<"embedding check passed\n";
        return 0;
    }
    catch( const std::exception& error )
    {
        std::cerr<<error.what()<<'\n';
        return 1;
    }
}
