#include "conversation_runtime.h"
#include "train.h"
#include <algorithm>
#include "cuda_memory.h"
#include "optimizer.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <nlohmann/json.hpp>
#include <charconv>

// Use an immutable bundle and prepared train split; never save or modify either.
int main(int argc, char** argv) {
    if(argc != 3 && argc != 4) {
        std::cerr << "Usage: cuda_memory_benchmark BUNDLE TRAIN_JSONL [BATCH_SIZE]\n";
        return 2;
    }
    try {
        auto bundle = g_loadConversation(argv[1]);
        auto& model = *bundle.spModel;
        model.setTraining(true);
        ConversationDataset dataset(g_readConversations(argv[2]), bundle.tokTokenizer,
                                    model.config().nContext);
        int batchSize = 32;
        if(argc == 4) {
            const std::string value(argv[3]);
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), batchSize);
            if(parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || batchSize <= 0)
                throw std::invalid_argument("Batch size must be positive");
        }
        constexpr int warmup = 3, measured = 10;
        if(dataset.size() < static_cast<std::size_t>(warmup + measured)*batchSize)
            throw std::runtime_error("Benchmark requires 13 full batches");
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
        const auto& config = model.config();
        nlohmann::json result = {
            {"allocator", after.pooled ? "pool" : "legacy"}, {"batch_size", batchSize},
            {"warmup_batches", warmup}, {"measured_batches", measured},
            {"seconds", seconds}, {"valid_tokens", tokens}, {"tokens_per_second", tokens/seconds},
            {"warmup_reserved_bytes", before.reservedBytes}, {"pool_used_bytes", after.usedBytes},
            {"pool_reserved_bytes", after.reservedBytes}, {"context", config.nContext},
            {"vocabulary", config.nVocabulary}, {"blocks", config.nBlocks},
            {"embedding", config.nEmbedding}, {"heads", config.nHeads}, {"hidden", config.nHidden},
            {"dropout", config.fDropout}, {"model_seed", config.nSeed}, {"shuffle_seed", 42},
            {"dtype", "float32"}, {"learning_rate", 0.0003}, {"clip_norm", 1.0},
            {"tokenizer", bundle.tokTokenizer.isSubword() ? "bpe" : "character"},
            {"bundle", argv[1]}, {"train_jsonl", argv[2]}
        };
        std::cout << result.dump() << '\n';
    } catch(const std::exception& error) {
        std::cerr << error.what() << '\n';
        // The caller retries a smaller batch only for a CUDA allocation OOM.
        return std::string(error.what()).find("out of memory") != std::string::npos ? 3 : 1;
    }
}
