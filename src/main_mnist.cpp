#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

#include "dataset_mnist.h"
#include "neuralnet_mnist.h"
#include "optimizer.h"
#include "trainer.h"

int main(int nArgCount,char** lpszArgs)
{
    try
    {
        const std::string c_strDataDirectory =nArgCount>=2
            ? lpszArgs[1]
            : "data/mnist";
        const std::filesystem::path c_pthModelDirectory =nArgCount>=3
            ? lpszArgs[2]
            : "models";

        MnistDataset mnsTrainDataset =MnistDataset::load(
            c_strDataDirectory + "/train-images-idx3-ubyte",
            c_strDataDirectory + "/train-labels-idx1-ubyte"
        );
        MnistDataset mnsTestDataset =MnistDataset::load(
            c_strDataDirectory + "/t10k-images-idx3-ubyte",
            c_strDataDirectory + "/t10k-labels-idx1-ubyte"
        );

        TrainConfig cfgConfig{
            .nEpochs        =5,
            .nBatchSize     =128,
            .nSeed          =42,
            .fLearningRate  =1.0e-6f,
            .fDropoutRate   =0.2f
        };

        MnistNeuralNet nntModel( cfgConfig.nSeed,cfgConfig.fDropoutRate );
        //nntModel.load( "mnist_model_batchnorm.bin" );
        Adam admOptimizer( &nntModel,cfgConfig.fLearningRate );

        train(
            nntModel,
            admOptimizer,
            mnsTrainDataset,
            cfgConfig
        );

        float fAccuracy =evaluate(
            nntModel,
            mnsTestDataset,
            cfgConfig.nBatchSize,
            countClassificationCorrect
        );

        printf(
            "test_accuracy=%.2f\n",
            fAccuracy*100.0f
        );

        nntModel.save( std::string(
            c_pthModelDirectory/"mnist_model_batchnorm.bin"
        ).c_str() );
    }
    catch( const std::exception& c_excError )
    {
        std::cerr << "error: " << c_excError.what() << '\n';
        return 1;
    }

    return 0;
}
