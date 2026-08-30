#include <cstdint>
#include <exception>
#include <iostream>
#include <string>

#include "mnist.h"
#include "neuralnet.h"
#include "optimizer.h"
#include "trainer.h"

int main(int nArgCount,char** lpszArgs)
{
    try
    {
        const std::string c_strDataDirectory =nArgCount>=2
            ? lpszArgs[1]
            : "data/mnist";

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
            .fLearningRate  =1.0e-3f,
            .fDropoutRate   =0.2f
        };

        NeuralNet nntModel( cfgConfig.nSeed,cfgConfig.fDropoutRate );
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

        nntModel.save( "mnist_dropout_model.bin" );
    }
    catch( const std::exception& c_excError )
    {
        std::cerr << "error: " << c_excError.what() << '\n';
        return 1;
    }

    return 0;
}
