#include "cuda_function_Add.h"
#include "cuda_function_BatchMatMul.h"
#include "cuda_function_LayerNorm.h"
#include "cuda_function_Mask.h"
#include "cuda_function_Permute.h"
#include "cuda_function_Reshape.h"
#include "cuda_function_Scale.h"
#include "cuda_function_Softmax.h"
#include "cuda_tensor.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition,const char* message)
{
    if( !condition )
    {
        throw std::runtime_error(message);
    }
}

void requireClose(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    const char* message,
    float tolerance =1.0e-4f
)
{
    require(actual.size()==expected.size(),message);
    for( std::size_t index=0;index<actual.size();++index )
    {
        if( std::fabs(actual[index]-expected[index])>tolerance )
        {
            throw std::runtime_error(message);
        }
    }
}

TensorPtr tensor(
    const std::vector<std::int64_t>& shape,
    const std::vector<float>& values
)
{
    auto result =std::make_shared<Tensor>(shape);
    result->_mData.copyFromHost(values.data(),values.size());
    return result;
}

cufMat gradient(
    const std::vector<std::int64_t>& shape,
    const std::vector<float>& values
)
{
    cufMat result(shape);
    result.copyFromHost(values.data(),values.size());
    return result;
}

void addAndScale()
{
    auto a =tensor({2,2},{1.0f,2.0f,3.0f,4.0f});
    auto b =tensor({2,2},{5.0f,6.0f,7.0f,8.0f});
    Add add;
    auto sum =add.forward({a,b})[0];
    requireClose(sum->_mData.toHost(),{6.0f,8.0f,10.0f,12.0f},"Add forward");

    cufMat sumGrad =gradient({2,2},{1.0f,2.0f,3.0f,4.0f});
    add.backward({&sumGrad},{a,b},{sum});
    requireClose(a->_mGrad.toHost(),sumGrad.toHost(),"Add A backward");
    requireClose(b->_mGrad.toHost(),sumGrad.toHost(),"Add B backward");

    Scale scale(-2.0f);
    auto scaled =scale.forward({a})[0];
    requireClose(scaled->_mData.toHost(),{-2.0f,-4.0f,-6.0f,-8.0f},"Scale forward");
    scale.backward({&sumGrad},{a},{scaled});
    requireClose(
        a->_mGrad.toHost(),
        {-1.0f,-2.0f,-3.0f,-4.0f},
        "Scale backward accumulation"
    );
}

void batchMatMul()
{
    auto a =tensor(
        {2,2,2},
        {1.0f,5.0f,2.0f,6.0f,3.0f,7.0f,4.0f,8.0f}
    );
    auto b =tensor(
        {2,2,2},
        {1.0f,2.0f,0.0f,0.0f,0.0f,0.0f,1.0f,2.0f}
    );
    BatchMatMul multiply;
    auto output =multiply.forward({a,b})[0];
    require(
        output->_mData.shape()==std::vector<std::int64_t>({2,2,2}),
        "BatchMatMul output shape"
    );
    requireClose(
        output->_mData.toHost(),
        {1.0f,10.0f,2.0f,12.0f,3.0f,14.0f,4.0f,16.0f},
        "BatchMatMul forward"
    );

    cufMat outputGrad =gradient(
        {2,2,2},
        {1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f}
    );
    multiply.backward({&outputGrad},{a,b},{output});
    requireClose(
        a->_mGrad.toHost(),
        {1.0f,2.0f,1.0f,2.0f,1.0f,2.0f,1.0f,2.0f},
        "BatchMatMul A backward"
    );
    requireClose(
        b->_mGrad.toHost(),
        {4.0f,12.0f,4.0f,12.0f,6.0f,14.0f,6.0f,14.0f},
        "BatchMatMul B backward"
    );
}

void layerNorm()
{
    Tensor gamma({2,1});
    Tensor beta({2,1});
    const std::vector<float> gammaValues{1.0f,2.0f};
    const std::vector<float> betaValues{0.5f,-0.5f};
    gamma._mData.copyFromHost(gammaValues.data(),gammaValues.size());
    beta._mData.copyFromHost(betaValues.data(),betaValues.size());

    auto input =tensor({2,2},{1.0f,3.0f,3.0f,7.0f});
    LayerNorm layerNorm(&gamma,&beta,1.0e-5f);
    auto output =layerNorm.forward({input})[0];
    requireClose(
        output->_mData.toHost(),
        {-0.499995f,-0.499999f,1.499990f,1.499998f},
        "LayerNorm forward"
    );

    cufMat outputGrad =gradient({2,2},{1.0f,1.0f,1.0f,1.0f});
    layerNorm.backward({&outputGrad},{input},{output});
    requireClose(gamma._mGrad.toHost(),{-1.999994f,1.999994f},"LayerNorm gamma");
    requireClose(beta._mGrad.toHost(),{2.0f,2.0f},"LayerNorm beta");
    const auto inputGrad =input->_mGrad.toHost();
    require(inputGrad[0]<0.0f&&inputGrad[2]>0.0f,"LayerNorm input gradient");
}

void mask()
{
    auto input =tensor(
        {2,3,2},
        {0.0f,1.0f,2.0f,3.0f,4.0f,5.0f,
         6.0f,7.0f,8.0f,9.0f,10.0f,11.0f}
    );
    Tensor maskTensor({2,3});
    const std::vector<float> maskValues{1.0f,0.0f,1.0f,0.0f,1.0f,0.0f};
    maskTensor._mData.copyFromHost(maskValues.data(),maskValues.size());
    Mask mask(&maskTensor);
    auto output =mask.forward({input})[0];
    const auto values =output->_mData.toHost();
    require(values[0]==0.0f&&values[1]==1.0f,"Mask kept value");
    require(std::isinf(values[2])&&values[2]<0.0f,"Mask removed value");
    require(values[4]==4.0f&&values[5]==5.0f,"Mask second kept value");
    require(values[8]==8.0f&&values[9]==9.0f,"Mask broadcast");

    cufMat outputGrad =gradient(
        {2,3,2},
        std::vector<float>(12,1.0f)
    );
    mask.backward({&outputGrad},{input},{output});
    requireClose(
        input->_mGrad.toHost(),
        {1.0f,1.0f,0.0f,0.0f,1.0f,1.0f,
         0.0f,0.0f,1.0f,1.0f,0.0f,0.0f},
        "Mask backward"
    );
}

void permuteAndReshape()
{
    auto input =tensor({2,3},{1.0f,2.0f,3.0f,4.0f,5.0f,6.0f});
    Permute permute({1,0});
    auto permuted =permute.forward({input})[0];
    require(
        permuted->_mData.shape()==std::vector<std::int64_t>({3,2}),
        "Permute shape"
    );
    requireClose(
        permuted->_mData.toHost(),
        {1.0f,4.0f,2.0f,5.0f,3.0f,6.0f},
        "Permute forward"
    );
    cufMat permutedGrad =gradient(
        {3,2},{1.0f,2.0f,3.0f,4.0f,5.0f,6.0f}
    );
    permute.backward({&permutedGrad},{input},{permuted});
    requireClose(
        input->_mGrad.toHost(),
        {1.0f,3.0f,5.0f,2.0f,4.0f,6.0f},
        "Permute backward"
    );

    Reshape reshape({3,-1});
    auto reshaped =reshape.forward({input})[0];
    require(
        reshaped->_mData.shape()==std::vector<std::int64_t>({3,2}),
        "Reshape inferred shape"
    );
    requireClose(reshaped->_mData.toHost(),input->_mData.toHost(),"Reshape forward");
    cufMat reshapeGrad =gradient(
        {3,2},{1.0f,1.0f,1.0f,1.0f,1.0f,1.0f}
    );
    reshape.backward({&reshapeGrad},{input},{reshaped});
    requireClose(
        input->_mGrad.toHost(),
        {2.0f,4.0f,6.0f,3.0f,5.0f,7.0f},
        "Reshape backward accumulation"
    );
}

void softmax()
{
    auto input =tensor({2,3},{1.0f,2.0f,3.0f,1.0f,1.0f,1.0f});
    Softmax softmax(1);
    auto output =softmax.forward({input})[0];
    const float denominator =
        std::exp(-2.0f)+std::exp(-1.0f)+1.0f;
    const std::vector<float> expected{
        std::exp(-2.0f)/denominator,
        std::exp(-1.0f)/denominator,
        1.0f/denominator,
        1.0f/3.0f,1.0f/3.0f,1.0f/3.0f
    };
    requireClose(output->_mData.toHost(),expected,"Softmax forward");

    const std::vector<float> gradValues{1.0f,0.0f,-1.0f,2.0f,0.0f,-2.0f};
    cufMat outputGrad =gradient({2,3},gradValues);
    softmax.backward({&outputGrad},{input},{output});
    std::vector<float> expectedGrad(6);
    for( std::size_t row=0;row<2;++row )
    {
        float dot =0.0f;
        for( std::size_t column=0;column<3;++column )
        {
            dot +=gradValues[row*3+column]*expected[row*3+column];
        }
        for( std::size_t column=0;column<3;++column )
        {
            const std::size_t index =row*3+column;
            expectedGrad[index] =expected[index]*(gradValues[index]-dot);
        }
    }
    requireClose(input->_mGrad.toHost(),expectedGrad,"Softmax backward");
}
} // namespace

int main()
{
    try
    {
        addAndScale();
        batchMatMul();
        layerNorm();
        mask();
        permuteAndReshape();
        softmax();
        std::cout<<"transformer functions check passed\n";
        return 0;
    }
    catch( const std::exception& error )
    {
        std::cerr<<error.what()<<'\n';
        return 1;
    }
}
