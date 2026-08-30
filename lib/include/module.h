#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cuda_function.h"
#include "cuda_tensor.h"

struct NamedTensor
{
    std::string strName;
    Tensor* lpTensor;
};

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
    void registerParameter(const std::string& c_strName,Tensor* lpTensor);
    void registerBuffer(const std::string& c_strName,Tensor* lpTensor);
    void registerModule(const std::string& c_strName,Module* lpModule);

private:
    struct NamedModule
    {
        std::string strName;
        Module* lpModule;
    };

    void validateRegistrationName(const std::string& c_strName) const;
    bool containsLocalName(const std::string& c_strName) const;

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

    virtual void save(const char* lpszFileName);
    virtual void load(const char* lpszFileName);
};
