#include <math.h>
#include "cuda_tensor.h"
#include "optimizer_adam.h"
#include "module.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <unordered_set>


// --------------------------
// AdamParams
// --------------------------
AdamParams::AdamParams(int nRows,int nCols)
    :AdamParams(std::vector<std::int64_t>{nRows,nCols})
{
}
AdamParams::AdamParams(const std::vector<std::int64_t>& shape)
    :_mM( shape ),
     _mV( shape )
{
    cuda_fill( _mM,0.0f );
    cuda_fill( _mV,0.0f );
}
AdamParams::~AdamParams()
{

}

// --------------------------
// Adam
// --------------------------
Adam::Adam(Model* lpModel,float fLearningRate)
    :Optimizer(lpModel,fLearningRate)
{

}
Adam::~Adam()
{

}
std::shared_ptr<OptimizerParams>
Adam::createOptimizerParams(Tensor* lpTensor)
{
    return std::make_shared<AdamParams>(lpTensor->_mData.shape());
}
void Adam::update_param(Tensor* lpTensor,OptimizerParams* lpOptimizerParams)
{
    // 更新手順はREADME参照
    AdamParams* lpAdamParams    =dynamic_cast<AdamParams*>(lpOptimizerParams);

    if( lpAdamParams==nullptr )
    {
        throw std::runtime_error(
            "Adam::update_param: invalid optimizer params"
        );
    }

    const float c_fBeta1      =0.9f;
    const float c_fBeta2      =0.999f;
    const float c_fEpsilon    =1.0e-8f;

    float fBeta1Correction =1.0f -powf(
        c_fBeta1,
        static_cast<float>(_nStep)
    );
    float fBeta2Correction =1.0f -powf(
        c_fBeta2,
        static_cast<float>(_nStep)
    );

    cuda_Adam_update(
        lpTensor->_mData,
        lpTensor->_mGrad,
        lpAdamParams->_mM,
        lpAdamParams->_mV,
        _fLearningRate,
        c_fBeta1,
        c_fBeta2,
        fBeta1Correction,
        fBeta2Correction,
        c_fEpsilon
    );
}

namespace {
constexpr char ADAM_MAGIC[8]={'A','I','C','P','P','A','D','M'};
constexpr std::uint32_t ADAM_VERSION=1;
template<class T> void writeValue(std::ostream& stream,const T& value)
{
    stream.write(reinterpret_cast<const char*>(&value),sizeof(value));
    if(!stream) throw std::runtime_error("Adam::saveState: failed to write state");
}
template<class T> T readValue(std::istream& stream)
{
    T value{};
    stream.read(reinterpret_cast<char*>(&value),sizeof(value));
    if(!stream) throw std::runtime_error("Adam::loadState: truncated state");
    return value;
}
}

void Adam::saveState(const std::string& c_strFileName) const
{
    const auto named=_lpModel->namedParameters();
    if(_lpParams.size()!=_spOptimizerParams.size()||named.size()!=_spOptimizerParams.size()||named.size()>UINT32_MAX)
        throw std::runtime_error("Adam::saveState: optimizer is not initialized");
    std::ofstream stream(c_strFileName,std::ios::binary|std::ios::trunc);
    if(!stream) throw std::runtime_error("Adam::saveState: cannot open state file");
    stream.write(ADAM_MAGIC,sizeof(ADAM_MAGIC));
    writeValue(stream,ADAM_VERSION); writeValue(stream,_fLearningRate); writeValue(stream,_nStep);
    writeValue(stream,static_cast<std::uint32_t>(named.size()));
    for(std::size_t i=0;i<named.size();++i)
    {
        const auto* state=dynamic_cast<const AdamParams*>(_spOptimizerParams[i].get());
        if(!state||named[i].lpTensor!=_lpParams[i]||named[i].strName.size()>MAX_NAME_LENGTH)
            throw std::runtime_error("Adam::saveState: invalid parameter state");
        const auto& shape=named[i].lpTensor->_mData.shape();
        writeValue(stream,static_cast<std::uint32_t>(named[i].strName.size()));
        stream.write(named[i].strName.data(),named[i].strName.size());
        writeValue(stream,static_cast<std::uint32_t>(shape.size()));
        for(auto extent:shape) writeValue(stream,extent);
        const auto first=state->_mM.toHost(),second=state->_mV.toHost();
        stream.write(reinterpret_cast<const char*>(first.data()),first.size()*sizeof(float));
        stream.write(reinterpret_cast<const char*>(second.data()),second.size()*sizeof(float));
        if(!stream) throw std::runtime_error("Adam::saveState: failed to write tensor state");
    }
}

void Adam::loadState(const std::string& c_strFileName)
{
    if(_lpParams.size()!=_spOptimizerParams.size())
        throw std::runtime_error("Adam::loadState: optimizer is not initialized");
    std::ifstream stream(c_strFileName,std::ios::binary);
    if(!stream) throw std::runtime_error("Adam::loadState: cannot open state file");
    char magic[sizeof(ADAM_MAGIC)]{}; stream.read(magic,sizeof(magic));
    if(!stream||std::memcmp(magic,ADAM_MAGIC,sizeof(magic))!=0||readValue<std::uint32_t>(stream)!=ADAM_VERSION)
        throw std::runtime_error("Adam::loadState: invalid state file");
    const float learningRate=readValue<float>(stream);
    const auto stepValue=readValue<std::uint64_t>(stream);
    const auto count=readValue<std::uint32_t>(stream);
    const auto named=_lpModel->namedParameters();
    if(!std::isfinite(learningRate)||learningRate<=0.0f||count!=named.size())
        throw std::runtime_error("Adam::loadState: incompatible state header");
    std::unordered_set<std::string> loaded;
    for(std::size_t i=0;i<named.size();++i)
    {
        const auto nameSize=readValue<std::uint32_t>(stream);
        if(nameSize==0||nameSize>MAX_NAME_LENGTH) throw std::runtime_error("Adam::loadState: invalid parameter name");
        std::string name(nameSize,'\0'); stream.read(name.data(),nameSize);
        const auto rank=readValue<std::uint32_t>(stream);
        std::vector<std::int64_t> shape(rank);
        for(auto& extent:shape) extent=readValue<std::int64_t>(stream);
        if(!stream||name!=named[i].strName||!loaded.insert(name).second||shape!=named[i].lpTensor->_mData.shape())
            throw std::runtime_error("Adam::loadState: parameter name or shape mismatch: "+name);
        auto* state=dynamic_cast<AdamParams*>(_spOptimizerParams[i].get());
        if(!state) throw std::runtime_error("Adam::loadState: invalid optimizer parameter");
        std::vector<float> first(state->_mM.numel()),second(state->_mV.numel());
        stream.read(reinterpret_cast<char*>(first.data()),first.size()*sizeof(float));
        stream.read(reinterpret_cast<char*>(second.data()),second.size()*sizeof(float));
        if(!stream) throw std::runtime_error("Adam::loadState: truncated tensor state");
        state->_mM.copyFromHost(first.data(),first.size()); state->_mV.copyFromHost(second.data(),second.size());
    }
    if(stream.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("Adam::loadState: trailing data");
    _fLearningRate=learningRate; _nStep=stepValue;
}

float Adam::learningRate() const noexcept {return _fLearningRate;}
std::uint64_t Adam::step() const noexcept {return _nStep;}
void Adam::setLearningRate(float fLearningRate)
{
    if(!std::isfinite(fLearningRate)||fLearningRate<=0.0f)
        throw std::invalid_argument("Adam learning rate must be finite and positive");
    _fLearningRate=fLearningRate;
}
