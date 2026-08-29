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

        constexpr std::uint32_t c_nSeed =42;
        constexpr float c_fLearningRate =1.0e-3f;

        NeuralNet nntModel( c_nSeed );
        Adam admOptimizer( &nntModel,c_fLearningRate );

        TrainConfig cfgConfig{
            ._nEpochs =5,
            ._nBatchSize =128,
            ._nSeed =c_nSeed
        };

        train(
            nntModel,
            admOptimizer,
            mnsTrainDataset,
            cfgConfig
        );

        float fAccuracy =evaluate(
            nntModel,
            mnsTestDataset,
            cfgConfig._nBatchSize,
            countClassificationCorrect
        );

        std::cout
            << "test_accuracy="
            << fAccuracy*100.0f
            << "%\n";

        nntModel.save( "mnist_model.bin" );
    }
    catch( const std::exception& c_excError )
    {
        std::cerr << "error: " << c_excError.what() << '\n';
        return 1;
    }

    return 0;
}
