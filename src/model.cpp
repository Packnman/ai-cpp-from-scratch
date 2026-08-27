#include "model.h"
#include <cmath>
#include <random>
#include <stdexcept>
#include "matrix.h"
namespace {
    void init_weight(Tensor& t,int fan,std::mt19937& rng)
    {
        Mat h(t._mData._nRows,t._mData._nCols);
        std::normal_distribution<float>d(0,std::sqrt(2.0f/fan));
        for(int i=0;i<t._mData._nRows*t._mData._nCols;++i)h._lpfHost[i]=d(rng);
        t._mData.download(h);
    }
}
MnistModel::MnistModel(std::uint32_t seed)
    :_w1(std::make_shared<Tensor>(256,784)),
     _b1(std::make_shared<Tensor>(256,1)),
     _w2(std::make_shared<Tensor>(128,256)),
     _b2(std::make_shared<Tensor>(128,1)),
     _w3(std::make_shared<Tensor>(10,128)),
     _b3(std::make_shared<Tensor>(10,1)),
     _linear1(_w1.get(),_b1.get()),
     _linear2(_w2.get(),_b2.get()),
     _linear3(_w3.get(),_b3.get())
{
    std::mt19937 rng(seed);
    init_weight(*_w1,784,rng);
    init_weight(*_w2,256,rng);
    init_weight(*_w3,128,rng);
    cuda_fill(_b1->_mData,0);
    cuda_fill(_b2->_mData,0);
    cuda_fill(_b3->_mData,0);
}
MnistModel::~MnistModel()
{

}
std::shared_ptr<Tensor> MnistModel::forward(std::vector<std::shared_ptr<Tensor>>& in)
{
    if(in.size()!=1||!in[0])
    {
        throw std::runtime_error(
            "MnistModel::forward: exactly one non-null input is required"
        );
    }
    if(in[0]->_mData._nRows!=784||in[0]->_mData._nCols<=0)
    {
        throw std::runtime_error(
            "MnistModel::forward: input must have shape 784 x batch"
        );
    }
    std::vector<std::shared_ptr<Tensor>>v{in[0]};
    auto x  =_linear1(v);
    v   ={x};
    x   =_relu1(v);
    v   ={x};
    x   =_linear2(v);
    v   ={x};
    x   =_relu2(v);
    v   ={x};

    return _linear3(v);
}
std::shared_ptr<Tensor> MnistModel::loss(
    const std::shared_ptr<Tensor>& input,
    const std::shared_ptr<Tensor>& target
)
{
    if(!input||!target)
    {
        throw std::runtime_error(
            "MnistModel::loss: input and target must be non-null"
        );
    }
    if(target->_mData._nRows!=10||target->_mData._nCols!=input->_mData._nCols)
    {
        throw std::runtime_error(
            "MnistModel::loss: target must have shape 10 x batch"
        );
    }
    std::vector<std::shared_ptr<Tensor>> in{input};
    auto logits =forward(in);
    std::vector<std::shared_ptr<Tensor>> args{logits,target};
    
    return _criterion(args);
}
std::vector<Tensor*> MnistModel::getParams()
{
    return {
        _w1.get(),
        _b1.get(),
        _w2.get(),
        _b2.get(),
        _w3.get(),
        _b3.get()
    };
}
