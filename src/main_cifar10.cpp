#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

#include "dataset_cifar10.h"
#include "neuralnet_cifar10.h"
#include "optimizer.h"
#include "trainer.h"

int main(int nArgCount,char** lpszArgs)
{
    try
    {
        const std::string c_strDataDirectory =nArgCount>=2
            ? lpszArgs[1]
            : "data/cifar-10-batches-bin";
        const std::filesystem::path c_pthModelDirectory =nArgCount>=3
            ? lpszArgs[2]
            : "models";

        Cifar10Dataset cifTrainDataset =
            Cifar10Dataset::loadTraining( c_strDataDirectory );
        Cifar10Dataset cifTestDataset =
            Cifar10Dataset::loadTest( c_strDataDirectory );

        TrainConfig cfgConfig{
            .nEpochs =20,
            .nBatchSize =128,
            .nSeed =42,
            .fLearningRate =1.0e-3f,
            .fDropoutRate =0.2f
        };

        NeuralNet_Cifar10 nntModel(
            cfgConfig.nSeed,
            cfgConfig.fDropoutRate
        );
        Adam admOptimizer( &nntModel,cfgConfig.fLearningRate );
        train(
            nntModel,
            admOptimizer,
            cifTrainDataset,
            cfgConfig
        );

        const float fAccuracy =evaluate(
            nntModel,
            cifTestDataset,
            cfgConfig.nBatchSize,
            countClassificationCorrect
        );
        printf( "test_accuracy=%.2f\n",fAccuracy*100.0f );

        nntModel.save(
            (c_pthModelDirectory/"cifar10_model.bin").string().c_str()
        );
    }
    catch( const std::exception& c_excError )
    {
        std::cerr << "error: " << c_excError.what() << '\n';
        return 1;
    }
    return 0;
}
