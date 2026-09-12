#include "model_transformer.h"
#include "optimizer.h"
#include <fstream>
#include <cmath>
#include <string>
#include <iomanip>
#include <iostream>

// Compare snapshots from separate processes with the two allocator settings.
int main(int argc, char** argv) {
    if(argc == 4 && std::string(argv[1]) == "--compare") {
        std::ifstream first(argv[2]), second(argv[3]);
        double a, b;
        std::size_t count = 0;
        while(first >> a) {
            if(!(second >> b) || !std::isfinite(a) || !std::isfinite(b) || std::abs(a-b) > 1e-6)
                return 1;
            ++count;
        }
        if(!first.eof() || (second >> b) || !second.eof() || count == 0) return 1;
        std::cout << "Compared " << count << " values within 1e-6\n";
        return 0;
    }
    if(argc != 2) return 2;
    try {
        std::ofstream output(argv[1]);
        output << std::setprecision(9);
        auto dump = [&](const cufMat& values) {
            for(float value : values.toHost()) output << value << '\n';
        };
        TransformerConfig config;
        config.nVocabulary = 12;
        config.nBlocks = 1;
        config.nEmbedding = 16;
        config.nHeads = 2;
        config.nHidden = 32;
        config.nContext = 8;
        config.nSeed = 42;
        config.fDropout = 0.1f;
        Transformer model(config);
        auto inputs = std::make_shared<cunMat>(4, 1);
        const std::int32_t ids[] = {2, 4, 7, 6};
        inputs->copyFromHost(ids, 4);
        cunMat targets(4, 1);
        const std::int32_t labels[] = {4, 7, 6, 3};
        targets.copyFromHost(labels, 4);
        Adam optimizer(&model, 0.001f);
        optimizer.init();
        for(const auto& parameter : model.namedParameters()) dump(parameter.lpTensor->_mData);
        dump(model.forward(inputs)->_mData);
        auto loss = model.loss(inputs, targets);
        dump(loss->_mData);
        auto later = model.forward(inputs);
        model.zero_grads();
        loss->backward();
        for(const auto& parameter : model.namedParameters()) dump(parameter.lpTensor->_mGrad);
        optimizer.update();
        for(const auto& parameter : model.namedParameters()) dump(parameter.lpTensor->_mData);
        if(!output) throw std::runtime_error("cannot write numeric snapshot");
    } catch(const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
