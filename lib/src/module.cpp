#include "module.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "matrix.h"

namespace {
    constexpr char MODEL_MAGIC[8] ={'A','I','C','P','P','M','D','L'};
    constexpr std::uint32_t MODEL_VERSION =1;
    constexpr std::uint32_t MAX_NAME_LENGTH =4096;

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

    void writeExact(FILE* lpFile,const void* lpData,std::size_t nSize)
    {
        if( (nSize>0)&&(fwrite(lpData,1,nSize,lpFile)!=nSize) )
        {
            throw std::runtime_error("Model::save: failed to write file");
        }
    }

    void readExact(FILE* lpFile,void* lpData,std::size_t nSize)
    {
        if( (nSize>0)&&(fread(lpData,1,nSize,lpFile)!=nSize) )
        {
            throw std::runtime_error("Model::load: truncated model file");
        }
    }

    template<class Type>
    void writeValue(FILE* lpFile,const Type& c_typValue)
    {
        writeExact( lpFile,&c_typValue,sizeof(Type) );
    }

    template<class Type>
    Type readValue(FILE* lpFile)
    {
        Type typValue{};
        readExact( lpFile,&typValue,sizeof(Type) );
        return typValue;
    }

    void appendWithPrefix(
        std::vector<NamedTensor>& nmtDestinations,
        const std::string& c_strPrefix,
        const std::vector<NamedTensor>& c_nmtSources
    )
    {
        for( const NamedTensor& c_nmtEntry : c_nmtSources )
        {
            nmtDestinations.push_back({
                c_strPrefix+c_nmtEntry._strName,
                c_nmtEntry._lpTensor
            });
        }
    }

    void zeroGradients(const std::vector<Tensor*>& c_lpParams)
    {
        for( Tensor* lpTensor : c_lpParams )
        {
            if( lpTensor!=nullptr )
            {
                cuda_fill( lpTensor->_mGrad,0.0f );
            }
        }
    }
}

Module::Module()
{
}

Module::~Module()
{
}

void Module::validateRegistrationName(const std::string& c_strName) const
{
    if( c_strName.empty() )
    {
        throw std::invalid_argument("Module: registration name must not be empty");
    }
    if( c_strName.find('.')!=std::string::npos )
    {
        throw std::invalid_argument("Module: registration name must not contain '.'");
    }
    if( containsLocalName(c_strName) )
    {
        throw std::invalid_argument("Module: duplicate registration name: "+c_strName);
    }
}

bool Module::containsLocalName(const std::string& c_strName) const
{
    auto fnTensorHasName =[&c_strName](const NamedTensor& c_nmtEntry)
    {
        return c_nmtEntry._strName==c_strName;
    };
    auto fnModuleHasName =[&c_strName](const NamedModule& c_nmmEntry)
    {
        return c_nmmEntry._strName==c_strName;
    };

    return
        std::any_of(_nmtParams.begin(),_nmtParams.end(),fnTensorHasName)||
        std::any_of(_nmtBuffers.begin(),_nmtBuffers.end(),fnTensorHasName)||
        std::any_of(_nmmModules.begin(),_nmmModules.end(),fnModuleHasName);
}

void Module::registerParameter(const std::string& c_strName,Tensor* lpTensor)
{
    validateRegistrationName( c_strName );
    if( lpTensor==nullptr )
    {
        throw std::invalid_argument("Module: parameter must not be null: "+c_strName);
    }
    _nmtParams.push_back({c_strName,lpTensor});
}

void Module::registerBuffer(const std::string& c_strName,Tensor* lpTensor)
{
    validateRegistrationName( c_strName );
    if( lpTensor==nullptr )
    {
        throw std::invalid_argument("Module: buffer must not be null: "+c_strName);
    }
    _nmtBuffers.push_back({c_strName,lpTensor});
}

void Module::registerModule(const std::string& c_strName,Module* lpModule)
{
    validateRegistrationName( c_strName );
    if( lpModule==nullptr )
    {
        throw std::invalid_argument("Module: child module must not be null: "+c_strName);
    }
    if( lpModule==this )
    {
        throw std::invalid_argument("Module: cannot register itself: "+c_strName);
    }
    _nmmModules.push_back({c_strName,lpModule});
}

std::vector<NamedTensor> Module::namedParameters() const
{
    std::vector<NamedTensor> nmtResults =_nmtParams;
    for( const NamedModule& c_nmmChild : _nmmModules )
    {
        appendWithPrefix(
            nmtResults,
            c_nmmChild._strName+".",
            c_nmmChild._lpModule->namedParameters()
        );
    }
    return nmtResults;
}

std::vector<NamedTensor> Module::namedBuffers() const
{
    std::vector<NamedTensor> nmtResults =_nmtBuffers;
    for( const NamedModule& c_nmmChild : _nmmModules )
    {
        appendWithPrefix(
            nmtResults,
            c_nmmChild._strName+".",
            c_nmmChild._lpModule->namedBuffers()
        );
    }
    return nmtResults;
}

std::vector<Tensor*> Module::getParams() const
{
    std::vector<NamedTensor> nmtParams =namedParameters();
    std::vector<Tensor*> lpResults;
    lpResults.reserve( nmtParams.size() );
    for( const NamedTensor& c_nmtParam : nmtParams )
    {
        lpResults.push_back( c_nmtParam._lpTensor );
    }
    return lpResults;
}

StateDict Module::stateDict() const
{
    StateDict nmtResults =namedParameters();
    std::vector<NamedTensor> nmtBuffers =namedBuffers();
    nmtResults.insert( nmtResults.end(),nmtBuffers.begin(),nmtBuffers.end() );
    return nmtResults;
}

void Module::zero_grads()
{
    zeroGradients( getParams() );
}

void Module::reset_state()
{
}

Model::Model()
    :Module()
{
}

Model::~Model()
{
}

void Model::save(const char* lpszFileName)
{
    if( lpszFileName==nullptr )
    {
        throw std::invalid_argument("Model::save: file name must not be null");
    }

    FilePtr filFile( fopen(lpszFileName,"wb") );
    if( filFile==nullptr )
    {
        throw std::runtime_error("Model::save: failed to open file");
    }

    StateDict nmtState =stateDict();
    if( nmtState.size()>static_cast<std::size_t>(UINT32_MAX) )
    {
        throw std::runtime_error("Model::save: too many state entries");
    }

    writeExact( filFile.get(),MODEL_MAGIC,sizeof(MODEL_MAGIC) );
    writeValue( filFile.get(),MODEL_VERSION );
    writeValue( filFile.get(),static_cast<std::uint32_t>(nmtState.size()) );

    for( const NamedTensor& c_nmtEntry : nmtState )
    {
        if( (c_nmtEntry._lpTensor==nullptr)||
            (c_nmtEntry._strName.size()>MAX_NAME_LENGTH) )
        {
            throw std::runtime_error(
                "Model::save: invalid state entry: "+c_nmtEntry._strName
            );
        }

        const std::uint32_t c_nNameLength =static_cast<std::uint32_t>(
            c_nmtEntry._strName.size()
        );
        const std::int32_t c_nRows =c_nmtEntry._lpTensor->_mData._nRows;
        const std::int32_t c_nCols =c_nmtEntry._lpTensor->_mData._nCols;
        Mat mHost( c_nRows,c_nCols );
        c_nmtEntry._lpTensor->_mData.upload( mHost );

        writeValue( filFile.get(),c_nNameLength );
        writeExact(
            filFile.get(),
            c_nmtEntry._strName.data(),
            c_nNameLength
        );
        writeValue( filFile.get(),c_nRows );
        writeValue( filFile.get(),c_nCols );
        writeExact(
            filFile.get(),
            mHost._lpfHost,
            static_cast<std::size_t>(c_nRows)*
                static_cast<std::size_t>(c_nCols)*sizeof(float)
        );
    }
}

void Model::load(const char* lpszFileName)
{
    if( lpszFileName==nullptr )
    {
        throw std::invalid_argument("Model::load: file name must not be null");
    }

    FilePtr filFile( fopen(lpszFileName,"rb") );
    if( filFile==nullptr )
    {
        throw std::runtime_error("Model::load: failed to open file");
    }

    char szMagic[sizeof(MODEL_MAGIC)]{};
    readExact( filFile.get(),szMagic,sizeof(szMagic) );
    if( std::memcmp(szMagic,MODEL_MAGIC,sizeof(MODEL_MAGIC))!=0 )
    {
        throw std::runtime_error("Model::load: invalid model file");
    }
    if( readValue<std::uint32_t>(filFile.get())!=MODEL_VERSION )
    {
        throw std::runtime_error("Model::load: unsupported model version");
    }

    StateDict nmtState =stateDict();
    std::unordered_map<std::string,Tensor*> umpDestinations;
    for( const NamedTensor& c_nmtEntry : nmtState )
    {
        if( (c_nmtEntry._lpTensor==nullptr)||
            !umpDestinations.emplace(
                c_nmtEntry._strName,
                c_nmtEntry._lpTensor
            ).second )
        {
            throw std::runtime_error(
                "Model::load: duplicate model state: "+c_nmtEntry._strName
            );
        }
    }

    const std::uint32_t c_nEntryCount =readValue<std::uint32_t>(filFile.get());
    if( c_nEntryCount!=nmtState.size() )
    {
        throw std::runtime_error("Model::load: state entry count mismatch");
    }

    std::unordered_set<std::string> ustLoadedNames;
    for( std::uint32_t nEntry=0;nEntry<c_nEntryCount;++nEntry )
    {
        const std::uint32_t c_nNameLength =readValue<std::uint32_t>(
            filFile.get()
        );
        if( (c_nNameLength==0)||(c_nNameLength>MAX_NAME_LENGTH) )
        {
            throw std::runtime_error("Model::load: invalid state name length");
        }

        std::string strName( c_nNameLength,'\0' );
        readExact( filFile.get(),strName.data(),c_nNameLength );
        const std::int32_t c_nRows =readValue<std::int32_t>(filFile.get());
        const std::int32_t c_nCols =readValue<std::int32_t>(filFile.get());

        auto itrDestination =umpDestinations.find( strName );
        if( itrDestination==umpDestinations.end() )
        {
            throw std::runtime_error("Model::load: unknown state entry: "+strName);
        }
        if( !ustLoadedNames.insert(strName).second )
        {
            throw std::runtime_error("Model::load: duplicate state entry: "+strName);
        }

        Tensor* lpTensor =itrDestination->second;
        if( (c_nRows!=lpTensor->_mData._nRows)||
            (c_nCols!=lpTensor->_mData._nCols) )
        {
            throw std::runtime_error("Model::load: state shape mismatch: "+strName);
        }

        Mat mHost( c_nRows,c_nCols );
        readExact(
            filFile.get(),
            mHost._lpfHost,
            static_cast<std::size_t>(c_nRows)*
                static_cast<std::size_t>(c_nCols)*sizeof(float)
        );
        lpTensor->_mData.download( mHost );
    }
}
