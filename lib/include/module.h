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
    Tensor*     lpTensor;
};

struct NamedModule
{
    std::string strName;
    Module*     lpModule;
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

// --------------------------
// Module
// --------------------------
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

private:
    std::vector<NamedTensor> _nmtParams;
    std::vector<NamedTensor> _nmtBuffers;
    std::vector<NamedModule> _nmmModules;
    bool _isTraining;

public: // propaties
    std::vector<NamedTensor>    namedParameters() const;
    std::vector<NamedTensor>    namedBuffers() const;
    std::vector<NamedTensor>    stateDict() const;
    std::vector<Tensor*>        getParams() const;
    void setTraining(bool isTraining);
    bool isTraining() const;

protected:
    void registerParameter(const std::string& c_strName,Tensor* lpTensor);
    void registerBuffer(const std::string& c_strName,Tensor* lpTensor);
    void registerModule(const std::string& c_strName,Module* lpModule);

private:

    void _validateRegistrationName(const std::string& c_strName) const;
    bool _containsLocalName(const std::string& c_strName) const;
    void _appendWithPrefix(
        std::vector<NamedTensor>& nmtDestinations,
        const std::string& c_strPrefix,
        const std::vector<NamedTensor>& c_nmtSources
    ) const;
    void _zeroGradients(const std::vector<Tensor*>& c_lpParams);

public:
    virtual void zero_grads();
    virtual void reset_state();
    virtual std::shared_ptr<Tensor> forward(
        std::vector<std::shared_ptr<Tensor>>& spmInputs
    ) =0;
};

// --------------------------
// Model
// --------------------------
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
