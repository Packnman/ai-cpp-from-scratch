#pragma once

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "cuda_function.h"
#include "cuda_tensor.h"

constexpr char MODEL_MAGIC[8] ={'A','I','C','P','P','M','D','L'};
constexpr std::uint32_t MODEL_VERSION =1;
constexpr std::uint32_t MAX_NAME_LENGTH =4096;

struct NamedTensor
{
    std::string strName;
    Tensor* lpTensor;
};

struct FileCloser
{
    void operator()(FILE* lpFile) const
    {
        if( lpFile!=nullptr )
        {
            fclose( lpFile );
        }
    }
};

using FilePtr =std::unique_ptr<FILE,FileCloser>;
using StateDict =std::vector<NamedTensor>;

// Tensorと子Moduleの所有権は派生クラスが持ち、Moduleは参照だけを登録する。
class Module
{
public:
    Module();
    virtual ~Module();

    Module(const Module&) =delete;
    Module& operator=(const Module&) =delete;
    Module(Module&&) =delete;
    Module& operator=(Module&&) =delete;

    std::vector<NamedTensor> namedParameters() const;
    std::vector<NamedTensor> namedBuffers() const;
    std::vector<Tensor*> getParams() const;
    StateDict stateDict() const;

    void setTraining(bool isTraining);
    bool isTraining() const;

    virtual void zero_grads();
    virtual void reset_state();
    virtual std::shared_ptr<Tensor> forward(
        std::vector<std::shared_ptr<Tensor>>& spmInputs
    ) =0;

protected:
    void __registerParameter(const std::string& c_strName,Tensor* lpTensor);
    void __registerBuffer(const std::string& c_strName,Tensor* lpTensor);
    void __registerModule(const std::string& c_strName,Module* lpModule);

private:
    struct NamedModule
    {
        std::string strName;
        Module* lpModule;
    };

    void _validateRegistrationName(const std::string& c_strName) const;
    bool _containsLocalName(const std::string& c_strName) const;
    void _appendWithPrefix(
        std::vector<NamedTensor>& nmtDestinations,
        const std::string& c_strPrefix,
        const std::vector<NamedTensor>& c_nmtSources
    ) const;
    void _zeroGradients(const std::vector<Tensor*>& c_lpParams);

    std::vector<NamedTensor> _nmtParams;
    std::vector<NamedTensor> _nmtBuffers;
    std::vector<NamedModule> _nmmModules;
    bool _isTraining;
};

class Model : public Module
{
public:
    Model();
    ~Model() override;

private:
    void _writeExact(FILE* filFile,const void* c_lpData,std::size_t nSize);
    void _readExact(FILE* filFile,void* lpData,std::size_t nSize);
    //
    template<class Type>
    void _writeValue(FILE* lpFile,const Type& c_typValue)
    {
        _writeExact( lpFile,&c_typValue,sizeof(Type) );
    }
    template<class Type>
    Type _readValue(FILE* lpFile)
    {
        Type typValue{};
        _readExact( lpFile,&typValue,sizeof(Type) );
        return typValue;
    }

public:
    virtual void save(const char* lpszFileName);
    virtual void load(const char* lpszFileName);
};

// 汎用Layer
class LayerConv2D : public Module
{
public:
    LayerConv2D(
        int nInputChannels,
        int nOutputChannels,
        int nInputHeight,
        int nInputWidth,
        int nKernelSize,
        int nStride =1,
        int nPadding =0
    );
    ~LayerConv2D() override;

private:
    std::shared_ptr<Tensor> _spmWeight;
    std::shared_ptr<Tensor> _spmBias;
    Conv2D _cnvConv2D;
    int _nOutputChannels;
    int _nOutputHeight;
    int _nOutputWidth;
    int _nFanIn;
public: // propaties
    int outputChannels() const;
    int outputHeight() const;
    int outputWidth() const;

private:
    int _checkedWeightRows(int nOutputChannels,int nInputChannels,int nKernelSize) const;
    int _checkedOutputChannels(int nOutputChannels) const;
    int _outputSize(int nInputSize,int nKernelSize,int nStride,int nPadding) const;
    int _fanIn(int nInputChannels,int nKernelSize) const;
public:
    void init(std::mt19937& rngRandom);
    //
    std::shared_ptr<Tensor> forward(
        std::vector<std::shared_ptr<Tensor>>& spmInputs
    ) override;
};
