#include <cstdint>
#include <exception>
#include <iostream>
#include <string>

#include "mnist.h"
#include "neuralnet.h"
#include "optimizer.h"
#include "trainer.h"

int main(int argc,char** argv)
{
    try
    {
        const std::string dataDirectory =argc>=2
            ? argv[1]
            : "data/mnist";

        MnistDataset trainDataset =MnistDataset::load(
            dataDirectory + "/train-images-idx3-ubyte",
            dataDirectory + "/train-labels-idx1-ubyte"
        );
        MnistDataset testDataset =MnistDataset::load(
            dataDirectory + "/t10k-images-idx3-ubyte",
            dataDirectory + "/t10k-labels-idx1-ubyte"
        );

        constexpr std::uint32_t seed =42;
        constexpr float learningRate =1.0e-3f;

        NeuralNet model( seed );
        Adam optimizer( &model,learningRate );

        TrainConfig config{
            .epochs =5,
            .batchSize =128,
            .seed =seed
        };

        train(
            model,
            optimizer,
            trainDataset,
            config
        );

        float accuracy =evaluate(
            model,
            testDataset,
            config.batchSize,
            countClassificationCorrect
        );

        std::cout
            << "test_accuracy="
            << accuracy*100.0f
            << "%\n";

        model.save( "mnist_model.bin" );
    }
    catch( const std::exception& error )
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
