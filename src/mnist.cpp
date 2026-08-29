#include "mnist.h"
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include "matrix.h"

namespace {
    // 関数オブジェクトの宣言（unique_ptrでのdeleteをカスタマイズした際の定石）
    struct FileCloser {
        void operator()(FILE* lpFile) const
        {
            if( lpFile!=nullptr )
            {
                fclose( lpFile );
            }
        }
    };

    // 通常はdeleteをカスタマイズしないため、
    // MyClass* lpValue =new MyClass(); 
    // std::unique_ptr<MyClass> upValue(lpValue);
    // std::unique_ptr<MyClass> upValue =lpValue; <- unique_ptrの生ポインタコンストラクタはexplicitなので不可
    // もしくは、
    // std::unique_ptr<MyClass> upValue(new MyClass());
    using FilePtr =std::unique_ptr<
        FILE,       // 対象
        FileCloser  // 削除方法
    >;

    std::uint32_t readUint32BigEndian(
        FILE* lpFile,
        const std::string& c_strPath
    )
    {
        std::uint8_t nBytes[4];
        std::size_t nRead =fread(
            nBytes,
            1,
            sizeof(nBytes),
            lpFile
        );
        if( nRead!=sizeof(nBytes) )
        {
            throw std::runtime_error(
                "MnistDataset::load: invalid IDX header: " + c_strPath
            );
        }

        return
            (static_cast<std::uint32_t>(nBytes[0])<<24)|
            (static_cast<std::uint32_t>(nBytes[1])<<16)|
            (static_cast<std::uint32_t>(nBytes[2])<<8) |
             static_cast<std::uint32_t>(nBytes[3]);
    }

    void readBytes(
        FILE* lpFile,
        std::vector<std::uint8_t>& nValues,
        const std::string& c_strPath
    )
    {
        std::size_t nRead =fread(
            nValues.data(),
            1,
            nValues.size(),
            lpFile
        );
        if( nRead!=nValues.size() )
        {
            throw std::runtime_error(
                "MnistDataset::load: truncated IDX data: " + c_strPath
            );
        }
    }
}

MnistDataset MnistDataset::load(
    const std::string& c_strImagePath,
    const std::string& c_strLabelPath
)
{
    FilePtr filImageFile(
        fopen( c_strImagePath.c_str(),"rb" )
    );
    if( filImageFile==nullptr )
    {
        throw std::runtime_error(
            "MnistDataset::load: failed to open image file: "
            + c_strImagePath
        );
    }

    FilePtr filLabelFile(
        fopen( c_strLabelPath.c_str(),"rb" )
    );
    if( filLabelFile==nullptr )
    {
        throw std::runtime_error(
            "MnistDataset::load: failed to open label file: "
            + c_strLabelPath
        );
    }

    std::uint32_t nImageMagic =readUint32BigEndian(
        filImageFile.get(),
        c_strImagePath
    );
    std::uint32_t nImageCount =readUint32BigEndian(
        filImageFile.get(),
        c_strImagePath
    );
    std::uint32_t nImageRows =readUint32BigEndian(
        filImageFile.get(),
        c_strImagePath
    );
    std::uint32_t nImageCols =readUint32BigEndian(
        filImageFile.get(),
        c_strImagePath
    );

    std::uint32_t nLabelMagic =readUint32BigEndian(
        filLabelFile.get(),
        c_strLabelPath
    );
    std::uint32_t nLabelCount =readUint32BigEndian(
        filLabelFile.get(),
        c_strLabelPath
    );

    if( nImageMagic!=2051 )
    {
        throw std::runtime_error(
            "MnistDataset::load: invalid image magic"
        );
    }
    if( nLabelMagic!=2049 )
    {
        throw std::runtime_error(
            "MnistDataset::load: invalid label magic"
        );
    }
    if( (nImageRows!=28)||(nImageCols!=28) )
    {
        throw std::runtime_error(
            "MnistDataset::load: image size must be 28 x 28"
        );
    }
    if( nImageCount==0 )
    {
        throw std::runtime_error(
            "MnistDataset::load: dataset is empty"
        );
    }
    if( nImageCount!=nLabelCount )
    {
        throw std::runtime_error(
            "MnistDataset::load: image and label count mismatch"
        );
    }

    constexpr std::size_t c_nImageSize =28*28;
    if( static_cast<std::size_t>(nImageCount)>
        std::numeric_limits<std::size_t>::max()/c_nImageSize )
    {
        throw std::runtime_error(
            "MnistDataset::load: image data is too large"
        );
    }

    MnistDataset mnsDataset;
    mnsDataset._nImages.resize(
        static_cast<std::size_t>(nImageCount)*c_nImageSize
    );
    mnsDataset._nLabels.resize(
        static_cast<std::size_t>(nLabelCount)
    );

    readBytes(
        filImageFile.get(),
        mnsDataset._nImages,
        c_strImagePath
    );
    readBytes(
        filLabelFile.get(),
        mnsDataset._nLabels,
        c_strLabelPath
    );

    for( std::uint8_t nLabel : mnsDataset._nLabels )
    {
        if( nLabel>=10 )
        {
            throw std::runtime_error(
                "MnistDataset::load: label must be between 0 and 9"
            );
        }
    }

    return mnsDataset;
}

std::size_t MnistDataset::size() const
{
    return _nLabels.size();
}

SupervisedBatch MnistDataset::makeBatch(
    const std::vector<std::size_t>& c_nIndices,
    std::size_t nBegin,
    std::size_t nCount
) const
{
    // 1. Tensor(784,count)とTensor(10,count)を生成する。
    // 2. 画像を0.0～1.0へ正規化して入力Tensorへ転送する。
    // 3. ラベルをone-hotへ変換してtarget Tensorへ転送する。
    if( nBegin+nCount>c_nIndices.size() )
    {
        throw std::runtime_error(
            "MnistDataset::makeBatch: invalid range"
        );
    }

    constexpr int c_nImageSize =28*28;
    constexpr int c_nClassCount =10;

    Mat mHostInput( c_nImageSize,static_cast<int>(nCount) );
    Mat mHostTarget( c_nClassCount,static_cast<int>(nCount) );

    // targetをすべて0にする
    for( int nBatch=0;nBatch<static_cast<int>(nCount);++nBatch )
    {
        for( int nLabel=0;nLabel<c_nClassCount;++nLabel )
        {
            mHostTarget(nLabel,nBatch) =0.0f;
        }
    }

    for( int nBatch=0;nBatch<static_cast<int>(nCount);++nBatch )
    {
        // シャッフル後の並びから画像番号を取得
        std::size_t nImageIndex =c_nIndices[nBegin+nBatch];

        // 画像の784ピクセルを1列へ入れる
        for( int nPixel=0;nPixel<c_nImageSize;++nPixel )
        {
            std::uint8_t nValue =_nImages[
                nImageIndex*c_nImageSize+nPixel
            ];
            mHostInput(nPixel,nBatch) =static_cast<float>(nValue)/255.0f;
        }

        // 正解ラベルをone-hotへ変換
        std::uint8_t nLabel =_nLabels[nImageIndex];
        mHostTarget(nLabel,nBatch) =1.0f;
    }

    std::shared_ptr<Tensor> spmInput =std::make_shared<Tensor>(
        c_nImageSize,
        static_cast<int>(nCount)
    );
    std::shared_ptr<Tensor> spmTarget =std::make_shared<Tensor>(
        c_nClassCount,
        static_cast<int>(nCount)
    );

    // CPUからGPUへ転送
    spmInput->_mData.download( mHostInput );
    spmTarget->_mData.download( mHostTarget );

    return {
        spmInput,
        spmTarget,
        nCount
    };
}

// --------------------------
// Mnist
// --------------------------
// 役割：
//  ・1バッチ内で「予測した数字」と「正解ラベル」が一致した画像数を返す
// --------------------------
std::size_t countClassificationCorrect(
    const std::shared_ptr<Tensor>& c_spmOutput, // NeuralNetの出力 logits
    const std::shared_ptr<Tensor>& c_spmTarget  // 正解one-hot
)
{
    if( (c_spmOutput==nullptr)||(c_spmTarget==nullptr) )
    {
        throw std::runtime_error(
            "countClassificationCorrect: tensor is null"
        );
    }
    if( (c_spmOutput->_mData._nRows!=c_spmTarget->_mData._nRows)||
        (c_spmOutput->_mData._nCols!=c_spmTarget->_mData._nCols)||
        (c_spmOutput->_mData._nRows<=0)||
        (c_spmOutput->_mData._nCols<=0) )
    {
        throw std::runtime_error(
            "countClassificationCorrect: tensor size mismatch"
        );
    }

    Mat mHostOutput(
        c_spmOutput->_mData._nRows,
        c_spmOutput->_mData._nCols
    );
    Mat mHostTarget(
        c_spmTarget->_mData._nRows,
        c_spmTarget->_mData._nCols
    );
    c_spmOutput->_mData.upload( mHostOutput );
    c_spmTarget->_mData.upload( mHostTarget );

    std::size_t nCorrect =0;

    for( int nBatch=0;nBatch<mHostOutput._nCols;++nBatch )
    {
        int nPredicted =0;
        int nExpected =0;

        // 各画像について、outputの値が最も大きい行番号を探します。
        for( int nRow=1;nRow<mHostOutput._nRows;++nRow )
        {
            if( mHostOutput(nRow,nBatch)>mHostOutput(nPredicted,nBatch) )
            {
                nPredicted =nRow;
            }
            if( mHostTarget(nRow,nBatch)>mHostTarget(nExpected,nBatch) )
            {
                nExpected =nRow;
            }
        }

        if( nPredicted==nExpected )
        {
            nCorrect++;
        }
    }

    return nCorrect;
}
