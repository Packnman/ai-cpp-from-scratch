#include "dataset_cifar10.h"

#include <cstdio>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>

#include "matrix.h"

namespace
{
constexpr std::size_t IMAGE_SIZE =32*32*3;
constexpr std::size_t PLANE_SIZE =32*32;
constexpr std::size_t RECORD_SIZE =1+IMAGE_SIZE;
constexpr int CLASS_COUNT =10;
constexpr float MEANS[3] ={0.4914f,0.4822f,0.4465f};
constexpr float STDS[3] ={0.2470f,0.2435f,0.2616f};

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

std::size_t fileSize(FILE* lpFile,const std::string& c_strPath)
{
    if( fseek(lpFile,0,SEEK_END)!=0 )
    {
        throw std::runtime_error(
            "Cifar10Dataset: failed to seek file: "+c_strPath
        );
    }
    const long nSize =ftell( lpFile );
    if( (nSize<0)||(fseek(lpFile,0,SEEK_SET)!=0) )
    {
        throw std::runtime_error(
            "Cifar10Dataset: failed to inspect file: "+c_strPath
        );
    }
    return static_cast<std::size_t>( nSize );
}
}

Cifar10Dataset Cifar10Dataset::loadTraining(
    const std::string& c_strDirectory
)
{
    return loadFiles(
        c_strDirectory,
        {
            "data_batch_1.bin",
            "data_batch_2.bin",
            "data_batch_3.bin",
            "data_batch_4.bin",
            "data_batch_5.bin"
        }
    );
}

Cifar10Dataset Cifar10Dataset::loadTest(
    const std::string& c_strDirectory
)
{
    return loadFiles( c_strDirectory,{"test_batch.bin"} );
}

Cifar10Dataset Cifar10Dataset::loadFiles(
    const std::string& c_strDirectory,
    const std::vector<std::string>& c_strFileNames
)
{
    Cifar10Dataset cifDataset;
    for( const std::string& c_strFileName : c_strFileNames )
    {
        cifDataset.appendFile(
            (std::filesystem::path(c_strDirectory)/c_strFileName).string()
        );
    }
    return cifDataset;
}

void Cifar10Dataset::appendFile(const std::string& c_strPath)
{
    FilePtr filFile( fopen(c_strPath.c_str(),"rb") );
    if( filFile==nullptr )
    {
        throw std::runtime_error(
            "Cifar10Dataset: failed to open file: "+c_strPath
        );
    }

    const std::size_t nSize =fileSize( filFile.get(),c_strPath );
    if( (nSize==0)||(nSize%RECORD_SIZE!=0) )
    {
        throw std::runtime_error(
            "Cifar10Dataset: file must contain complete records: "+c_strPath
        );
    }
    const std::size_t nRecordCount =nSize/RECORD_SIZE;
    if( nRecordCount>std::numeric_limits<std::size_t>::max()/IMAGE_SIZE||
        _nImages.size()>std::numeric_limits<std::size_t>::max()-
            nRecordCount*IMAGE_SIZE )
    {
        throw std::runtime_error("Cifar10Dataset: dataset is too large");
    }

    std::vector<std::uint8_t> nRecord( RECORD_SIZE );
    _nImages.reserve( _nImages.size()+nRecordCount*IMAGE_SIZE );
    _nLabels.reserve( _nLabels.size()+nRecordCount );
    for( std::size_t nRecordIndex=0;
         nRecordIndex<nRecordCount;
         ++nRecordIndex )
    {
        if( fread(nRecord.data(),1,nRecord.size(),filFile.get())!=
            nRecord.size() )
        {
            throw std::runtime_error(
                "Cifar10Dataset: truncated record: "+c_strPath
            );
        }
        if( nRecord[0]>=CLASS_COUNT )
        {
            throw std::runtime_error(
                "Cifar10Dataset: label must be between 0 and 9: "+c_strPath
            );
        }
        _nLabels.push_back( nRecord[0] );
        _nImages.insert(
            _nImages.end(),
            nRecord.begin()+1,
            nRecord.end()
        );
    }
}

std::size_t Cifar10Dataset::size() const
{
    return _nLabels.size();
}

SupervisedBatch Cifar10Dataset::makeBatch(
    const std::vector<std::size_t>& c_nIndices,
    std::size_t nBegin,
    std::size_t nCount
) const
{
    if( (nCount==0)||(nCount>static_cast<std::size_t>(
            std::numeric_limits<int>::max()))||
        (nBegin>c_nIndices.size())||
        (nCount>c_nIndices.size()-nBegin) )
    {
        throw std::runtime_error(
            "Cifar10Dataset::makeBatch: invalid range"
        );
    }

    Mat mHostInput(
        static_cast<int>(IMAGE_SIZE),
        static_cast<int>(nCount)
    );
    Mat mHostTarget( CLASS_COUNT,static_cast<int>(nCount) );
    for( int nBatch=0;nBatch<static_cast<int>(nCount);++nBatch )
    {
        for( int nClass=0;nClass<CLASS_COUNT;++nClass )
        {
            mHostTarget(nClass,nBatch) =0.0f;
        }

        const std::size_t nImageIndex =c_nIndices[nBegin+nBatch];
        if( nImageIndex>=size() )
        {
            throw std::runtime_error(
                "Cifar10Dataset::makeBatch: index out of range"
            );
        }
        for( int nY=0;nY<32;++nY )
        {
            for( int nX=0;nX<32;++nX )
            {
                for( int nChannel=0;nChannel<3;++nChannel )
                {
                    const std::size_t nPlanar =
                        nImageIndex*IMAGE_SIZE+
                        static_cast<std::size_t>(nChannel)*PLANE_SIZE+
                        static_cast<std::size_t>(nY)*32+nX;
                    const int nHwc =(nY*32+nX)*3+nChannel;
                    const float fPixel =
                        static_cast<float>(_nImages[nPlanar])/255.0f;
                    mHostInput(nHwc,nBatch) =
                        (fPixel-MEANS[nChannel])/STDS[nChannel];
                }
            }
        }
        mHostTarget(_nLabels[nImageIndex],nBatch) =1.0f;
    }

    auto spmInput =std::make_shared<Tensor>(
        static_cast<int>(IMAGE_SIZE),
        static_cast<int>(nCount)
    );
    auto spmTarget =std::make_shared<Tensor>(
        CLASS_COUNT,
        static_cast<int>(nCount)
    );
    spmInput->_mData.download( mHostInput );
    spmTarget->_mData.download( mHostTarget );

    return {spmInput,spmTarget,nCount};
}
