#include "mnist.h"
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include "matrix.h"

namespace {
    // 関数オブジェクトの宣言（unique_ptrでのdeleteをカスタマイズした際の定石）
    struct FileCloser {
        void operator()(FILE* fp) const
        {
            if( fp!=nullptr )
            {
                fclose( fp );
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
        FILE* fp,
        const std::string& path
    )
    {
        std::uint8_t bytes[4];
        std::size_t nRead =fread(
            bytes,
            1,
            sizeof(bytes),
            fp
        );
        if( nRead!=sizeof(bytes) )
        {
            throw std::runtime_error(
                "MnistDataset::load: invalid IDX header: " + path
            );
        }

        return
            (static_cast<std::uint32_t>(bytes[0])<<24)|
            (static_cast<std::uint32_t>(bytes[1])<<16)|
            (static_cast<std::uint32_t>(bytes[2])<<8) |
             static_cast<std::uint32_t>(bytes[3]);
    }

    void readBytes(
        FILE* fp,
        std::vector<std::uint8_t>& values,
        const std::string& path
    )
    {
        std::size_t nRead =fread(
            values.data(),
            1,
            values.size(),
            fp
        );
        if( nRead!=values.size() )
        {
            throw std::runtime_error(
                "MnistDataset::load: truncated IDX data: " + path
            );
        }
    }
}

MnistDataset MnistDataset::load(
    const std::string& imagePath,
    const std::string& labelPath
)
{
    FilePtr imageFile(
        fopen( imagePath.c_str(),"rb" )
    );
    if( imageFile==nullptr )
    {
        throw std::runtime_error(
            "MnistDataset::load: failed to open image file: "
            + imagePath
        );
    }

    FilePtr labelFile(
        fopen( labelPath.c_str(),"rb" )
    );
    if( labelFile==nullptr )
    {
        throw std::runtime_error(
            "MnistDataset::load: failed to open label file: "
            + labelPath
        );
    }

    std::uint32_t imageMagic =readUint32BigEndian(
        imageFile.get(),
        imagePath
    );
    std::uint32_t imageCount =readUint32BigEndian(
        imageFile.get(),
        imagePath
    );
    std::uint32_t imageRows =readUint32BigEndian(
        imageFile.get(),
        imagePath
    );
    std::uint32_t imageCols =readUint32BigEndian(
        imageFile.get(),
        imagePath
    );

    std::uint32_t labelMagic =readUint32BigEndian(
        labelFile.get(),
        labelPath
    );
    std::uint32_t labelCount =readUint32BigEndian(
        labelFile.get(),
        labelPath
    );

    if( imageMagic!=2051 )
    {
        throw std::runtime_error(
            "MnistDataset::load: invalid image magic"
        );
    }
    if( labelMagic!=2049 )
    {
        throw std::runtime_error(
            "MnistDataset::load: invalid label magic"
        );
    }
    if( (imageRows!=28)||(imageCols!=28) )
    {
        throw std::runtime_error(
            "MnistDataset::load: image size must be 28 x 28"
        );
    }
    if( imageCount==0 )
    {
        throw std::runtime_error(
            "MnistDataset::load: dataset is empty"
        );
    }
    if( imageCount!=labelCount )
    {
        throw std::runtime_error(
            "MnistDataset::load: image and label count mismatch"
        );
    }

    constexpr std::size_t imageSize =28*28;
    if( static_cast<std::size_t>(imageCount)>
        std::numeric_limits<std::size_t>::max()/imageSize )
    {
        throw std::runtime_error(
            "MnistDataset::load: image data is too large"
        );
    }

    MnistDataset dataset;
    dataset._images.resize(
        static_cast<std::size_t>(imageCount)*imageSize
    );
    dataset._labels.resize(
        static_cast<std::size_t>(labelCount)
    );

    readBytes(
        imageFile.get(),
        dataset._images,
        imagePath
    );
    readBytes(
        labelFile.get(),
        dataset._labels,
        labelPath
    );

    for( std::uint8_t label : dataset._labels )
    {
        if( label>=10 )
        {
            throw std::runtime_error(
                "MnistDataset::load: label must be between 0 and 9"
            );
        }
    }

    return dataset;
}

std::size_t MnistDataset::size() const
{
    return _labels.size();
}

SupervisedBatch MnistDataset::makeBatch(
    const std::vector<std::size_t>& indices,
    std::size_t begin,
    std::size_t count
) const
{
    // 1. Tensor(784,count)とTensor(10,count)を生成する。
    // 2. 画像を0.0～1.0へ正規化して入力Tensorへ転送する。
    // 3. ラベルをone-hotへ変換してtarget Tensorへ転送する。
    if( begin+count>indices.size() )
    {
        throw std::runtime_error(
            "MnistDataset::makeBatch: invalid range"
        );
    }

    constexpr int imageSize     =28*28;
    constexpr int classCount    =10;

    Mat hostInput(  imageSize ,static_cast<int>(count) );
    Mat hostTarget( classCount,static_cast<int>(count) );

    // targetをすべて0にする
    for( int batch=0;batch<static_cast<int>(count);batch++ )
    {
        for( int label=0;label<classCount;label++ )
        {
            hostTarget(label,batch) =0.0f;
        }
    }

    for( int batch=0;batch<static_cast<int>(count);batch++ )
    {
        // シャッフル後の並びから画像番号を取得
        std::size_t imageIndex  =indices[begin+batch];

        // 画像の784ピクセルを1列へ入れる
        for( int pixel=0;pixel<imageSize;pixel++ )
        {
            std::uint8_t value      =_images[imageIndex*imageSize+pixel];
            hostInput(pixel,batch)  =static_cast<float>(value)/255.0f;
        }

        // 正解ラベルをone-hotへ変換
        std::uint8_t label      =_labels[imageIndex];
        hostTarget(label,batch) =1.0f;
    }

    std::shared_ptr<Tensor> input   =std::make_shared<Tensor>( imageSize ,static_cast<int>(count) );
    std::shared_ptr<Tensor> target  =std::make_shared<Tensor>( classCount,static_cast<int>(count) );

    // CPUからGPUへ転送
    input->_mData.download(hostInput);
    target->_mData.download(hostTarget);

    return {
        input,
        target,
        count
    };
}

// --------------------------
// Mnist
// --------------------------
// 役割：
//  ・1バッチ内で「予測した数字」と「正解ラベル」が一致した画像数を返す
// --------------------------
std::size_t countClassificationCorrect(
    const std::shared_ptr<Tensor>& output,  // NeuralNetの出力 logits : 10 x batch
    const std::shared_ptr<Tensor>& target   // 正解one-hot : 10 x batch
)
{
    if( (output==nullptr)||(target==nullptr) )
    {
        throw std::runtime_error(
            "countClassificationCorrect: tensor is null"
        );
    }
    if( (output->_mData._nRows!=target->_mData._nRows)||
        (output->_mData._nCols!=target->_mData._nCols)||
        (output->_mData._nRows<=0)||
        (output->_mData._nCols<=0) )
    {
        throw std::runtime_error(
            "countClassificationCorrect: tensor size mismatch"
        );
    }

    Mat hostOutput(
        output->_mData._nRows,
        output->_mData._nCols
    );
    Mat hostTarget(
        target->_mData._nRows,
        target->_mData._nCols
    );
    output->_mData.upload( hostOutput );
    target->_mData.upload( hostTarget );

    std::size_t nCorrect =0;

    for( int batch=0;batch<hostOutput._nCols;batch++ )
    {
        int predicted   =0;
        int expected    =0;

        // 各画像について、outputの値が最も大きい行番号を探します。
        for( int row=1;row<hostOutput._nRows;row++ )
        {
            if( hostOutput(row,batch)>hostOutput(predicted,batch) )
            {
                predicted   =row;
            }
            if( hostTarget(row,batch)>hostTarget(expected,batch) )
            {
                expected    =row;
            }
        }

        if( predicted==expected )
        {
            nCorrect++;
        }
    }

    return nCorrect;
}
