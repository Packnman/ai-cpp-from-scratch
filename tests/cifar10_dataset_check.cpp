#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dataset_cifar10.h"
#include "matrix.h"

namespace
{
void require(bool isCondition,const std::string& c_strMessage)
{
    if( !isCondition )
    {
        throw std::runtime_error( c_strMessage );
    }
}

void writeRecord(
    const std::filesystem::path& c_pthPath,
    std::uint8_t nLabel,
    std::uint8_t nRed,
    std::uint8_t nGreen,
    std::uint8_t nBlue
)
{
    std::ofstream ofsFile( c_pthPath,std::ios::binary );
    ofsFile.put( static_cast<char>(nLabel) );
    for( int nChannel=0;nChannel<3;++nChannel )
    {
        const std::uint8_t nValue =
            nChannel==0 ? nRed : (nChannel==1 ? nGreen : nBlue);
        for( int nPixel=0;nPixel<32*32;++nPixel )
        {
            ofsFile.put( static_cast<char>(nValue) );
        }
    }
}

template<class FunctionType>
void requireThrows(FunctionType fnFunction,const std::string& c_strMessage)
{
    bool isThrown =false;
    try
    {
        fnFunction();
    }
    catch( const std::exception& )
    {
        isThrown =true;
    }
    require( isThrown,c_strMessage );
}

void checkBatch(const std::filesystem::path& c_pthRoot)
{
    for( int nFile=1;nFile<=5;++nFile )
    {
        writeRecord(
            c_pthRoot/("data_batch_"+std::to_string(nFile)+".bin"),
            static_cast<std::uint8_t>(nFile),
            0,
            128,
            255
        );
    }
    Cifar10Dataset cifDataset =
        Cifar10Dataset::loadTraining( c_pthRoot.string() );
    require( cifDataset.size()==5,"training files were not combined" );

    const std::vector<std::size_t> nIndices{4,0};
    SupervisedBatch batBatch =cifDataset.makeBatch( nIndices,0,2 );
    require(
        (batBatch.spmInput->_mData._nRows==3072)&&
        (batBatch.spmInput->_mData._nCols==2)&&
        (batBatch.spmTarget->_mData._nRows==10)&&
        (batBatch.nSize==2),
        "batch shape mismatch"
    );

    Mat mInput( 3072,2 );
    Mat mTarget( 10,2 );
    batBatch.spmInput->_mData.upload( mInput );
    batBatch.spmTarget->_mData.upload( mTarget );
    const float fExpected[3]{
        (0.0f-0.4914f)/0.2470f,
        (128.0f/255.0f-0.4822f)/0.2435f,
        (1.0f-0.4465f)/0.2616f
    };
    for( int nBatch=0;nBatch<2;++nBatch )
    {
        for( int nChannel=0;nChannel<3;++nChannel )
        {
            require(
                std::fabs(mInput(nChannel,nBatch)-fExpected[nChannel])<
                    1.0e-5f,
                "planar to HWC conversion or normalization mismatch"
            );
            require(
                std::fabs(mInput((17*32+9)*3+nChannel,nBatch)-
                    fExpected[nChannel])<1.0e-5f,
                "HWC pixel mismatch"
            );
        }
    }
    require(
        (mTarget(5,0)==1.0f)&&(mTarget(1,1)==1.0f),
        "reordered one-hot target mismatch"
    );

    requireThrows(
        [&](){cifDataset.makeBatch(nIndices,1,2);},
        "invalid batch range was accepted"
    );
    requireThrows(
        [&](){cifDataset.makeBatch({5},0,1);},
        "out-of-range dataset index was accepted"
    );
    requireThrows(
        [&](){cifDataset.makeBatch(nIndices,0,0);},
        "empty batch was accepted"
    );
}

void checkInvalidFiles(const std::filesystem::path& c_pthRoot)
{
    requireThrows(
        [&](){Cifar10Dataset::loadTest((c_pthRoot/"missing").string());},
        "missing file was accepted"
    );

    const auto pthEmpty =c_pthRoot/"empty";
    std::filesystem::create_directory( pthEmpty );
    std::ofstream( pthEmpty/"test_batch.bin",std::ios::binary );
    requireThrows(
        [&](){Cifar10Dataset::loadTest(pthEmpty.string());},
        "empty file was accepted"
    );

    const auto pthPartial =c_pthRoot/"partial";
    std::filesystem::create_directory( pthPartial );
    {
        std::ofstream ofsFile( pthPartial/"test_batch.bin",std::ios::binary );
        ofsFile.put( 1 );
    }
    requireThrows(
        [&](){Cifar10Dataset::loadTest(pthPartial.string());},
        "partial record was accepted"
    );

    const auto pthLabel =c_pthRoot/"label";
    std::filesystem::create_directory( pthLabel );
    writeRecord( pthLabel/"test_batch.bin",10,0,0,0 );
    requireThrows(
        [&](){Cifar10Dataset::loadTest(pthLabel.string());},
        "invalid label was accepted"
    );
}
}

int main()
{
    const std::filesystem::path c_pthRoot =
        std::filesystem::temp_directory_path()/"ai_cpp_cifar10_dataset_check";
    try
    {
        std::filesystem::remove_all( c_pthRoot );
        std::filesystem::create_directories( c_pthRoot );
        checkBatch( c_pthRoot );
        checkInvalidFiles( c_pthRoot );
        std::filesystem::remove_all( c_pthRoot );
        std::cout << "cifar10 dataset check passed\n";
        return 0;
    }
    catch( const std::exception& c_excError )
    {
        std::filesystem::remove_all( c_pthRoot );
        std::cerr << c_excError.what() << '\n';
        return 1;
    }
}
