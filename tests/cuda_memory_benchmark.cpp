#include "conversation_runtime.h"
#include "train.h"
#include <algorithm>
#include "cuda_memory.h"
#include "optimizer.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>

// Use an immutable bundle and prepared train split; never save or modify either.
int main(int argc, char** argv) {
    if(argc != 3) {
        std::cerr << "Usage: cuda_memory_benchmark BUNDLE TRAIN_JSONL\n";
        return 2;
    }
    try {
        auto bundle = g_loadConversation(argv[1]);
        auto& model = *bundle.spModel;
        model.setTraining(true);
        ConversationDataset dataset(g_readConversations(argv[2]), bundle.tokTokenizer,
                                    model.config().nContext);
        constexpr int batchSize = 32, warmup = 3, measured = 10;
        if(dataset.size() < (warmup + measured)*batchSize)
            throw std::runtime_error("Benchmark requires at least 416 windows");
        std::vector<std::size_t> order(dataset.size());
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 random(42);
        std::shuffle(order.begin(), order.end(), random);
        Adam optimizer(&model, 0.0003f);
        optimizer.init();
        auto step = [&](int index) {
            const auto batch = dataset.batch(order, index*batchSize, batchSize);
            auto inputs = std::make_shared<cunMat>(batch.nSequence, batch.nBatch);
            cunMat targets(batch.nSequence, batch.nBatch);
            inputs->copyFromHost(batch.nInputs.data(), batch.nInputs.size());
            targets.copyFromHost(batch.nTargets.data(), batch.nTargets.size());
            model.zero_grads();
            auto loss = model.loss(inputs, targets);
            if(!std::isfinite(loss->_mData.toHost()[0])) throw std::runtime_error("Nonfinite loss");
            loss->backward();
            ClipGradients(model, 1.0f);
            optimizer.update();
            return batch.nValid;
        };
        for(int i = 0; i < warmup; ++i) step(i);
        const auto before = cu_memory::statistics();
        const auto start = std::chrono::steady_clock::now();
        std::size_t tokens = 0;
        for(int i = warmup; i < warmup + measured; ++i) tokens += step(i);
        const auto after = cu_memory::statistics();
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::cout << "{\"allocator\":\"" << (after.pooled ? "pool" : "legacy")
                  << "\",\"batch_size\":32,\"warmup_batches\":3,\"measured_batches\":10"
                  << ",\"seconds\":" << seconds << ",\"tokens_per_second\":" << tokens/seconds
                  << ",\"warmup_reserved_bytes\":" << before.reservedBytes
                  << ",\"pool_used_bytes\":" << after.usedBytes
                  << ",\"pool_reserved_bytes\":" << after.reservedBytes << "}\n";
    } catch(const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
