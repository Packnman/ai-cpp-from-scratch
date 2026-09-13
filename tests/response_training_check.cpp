#include "optimizer.h"
#include "train.h"
#include "validation.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {
void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
} // namespace

int main() {
    const std::vector<std::pair<std::string, std::string>> qa = {
        {"赤は", "あか"}, {"青は", "あお"}, {"猫は", "ねこ"}, {"犬は", "いぬ"},
        {"春は", "はる"}, {"夏は", "なつ"}, {"海は", "うみ"}, {"山は", "やま"}};
    std::vector<Conversation> conversations;
    std::string vocabulary;
    for (std::size_t index = 0; index < qa.size(); ++index) {
        conversations.push_back(
            {static_cast<std::int64_t>(index),
             {{0, qa[index].first}, {1, qa[index].second}}});
        vocabulary += qa[index].first + qa[index].second;
    }
    TokenConversation tokenizer(vocabulary);
    ConversationDataset data(conversations, tokenizer, 16,
                             ConversationLossTarget::Response);
    require(data.size() == qa.size() && data.excludedResponses() == 0,
            "Every B response must produce one sample");

    std::vector<std::size_t> order(data.size());
    std::iota(order.begin(), order.end(), 0);
    const auto first = data.batch(order, 0, 1);
    TokenIds activeTargets;
    for (int target : first.nTargets)
        if (target != TokenConversation::PAD)
            activeTargets.push_back(target);
    auto answer = tokenizer.encode(qa[0].second);
    answer.push_back(TokenConversation::UTTERANCE_END);
    require(activeTargets == answer,
            "Only response text and its terminator may have loss");
    const auto question = tokenizer.encode(qa[0].first);
    require(std::search(first.nInputs.begin(), first.nInputs.end(),
                        question.begin(),
                        question.end()) != first.nInputs.end(),
            "Question and response must share a sample");

    ConversationDataset tooLong({{99, {{0, "赤は赤は"}, {1, "あかあか"}}}},
                                tokenizer, 4, ConversationLossTarget::Response);
    require(tooLong.size() == 0 && tooLong.excludedResponses() == 1,
            "An indivisible overlong Q/A pair must be excluded");

    ConversationDataset history(
        {{100, {{0, "赤は"}, {1, "あか"}, {0, "青は"}, {1, "あお"}}}},
        tokenizer, 32, ConversationLossTarget::Response);
    const auto second = history.batch({1}, 0, 1);
    const auto oldAnswer = tokenizer.encode("あか");
    require(std::search(second.nInputs.begin(), second.nInputs.end(),
                        oldAnswer.begin(),
                        oldAnswer.end()) != second.nInputs.end(),
            "Older complete turns should be prepended when they fit");

    TransformerConfig config;
    config.nVocabulary = tokenizer.vocabSize();
    config.nBlocks = 2;
    config.nEmbedding = 32;
    config.nHeads = 4;
    config.nHidden = 64;
    config.nContext = 16;
    config.fDropout = 0.0f;
    config.nSeed = 123;
    Transformer model(config);
    Adam optimizer(&model, 0.02f);
    optimizer.init();
    const auto batch = data.batch(order, 0, static_cast<int>(order.size()));
    auto inputs = std::make_shared<cunMat>(batch.nSequence, batch.nBatch);
    inputs->copyFromHost(batch.nInputs.data(), batch.nInputs.size());
    cunMat targets(batch.nSequence, batch.nBatch);
    targets.copyFromHost(batch.nTargets.data(), batch.nTargets.size());
    const float initial = model.loss(inputs, targets)->_mData.toHost()[0];
    for (int step = 0; step < 500; ++step) {
        model.zero_grads();
        auto loss = model.loss(inputs, targets);
        loss->backward();
        ClipGradients(model, 1.0f);
        optimizer.update();
    }
    model.setTraining(false);
    const float final = model.loss(inputs, targets)->_mData.toHost()[0];
    require(std::isfinite(final) && final < 0.05f && final < initial * 0.05f,
            "Response-only fixture loss must clearly decrease");

    ConfigGeneration generation;
    generation.nTopK = 1;
    generation.nMaxTokens = 8;
    std::mt19937 random(123);
    for (const auto &[prompt, expectedText] : qa) {
        TokenIds input = {TokenConversation::BEGIN,
                          TokenConversation::SPEAKER_A};
        const auto encoded = tokenizer.encode(prompt);
        input.insert(input.end(), encoded.begin(), encoded.end());
        input.push_back(TokenConversation::UTTERANCE_END);
        input.push_back(TokenConversation::SPEAKER_B);
        auto expected = tokenizer.encode(expectedText);
        expected.push_back(TokenConversation::UTTERANCE_END);
        require(Generate(model, tokenizer, input, random, generation) ==
                    expected,
                "Greedy generation must reproduce every memorized answer");
    }
    std::cout << "response-only Q/A overfit " << initial << " -> " << final
              << '\n';
    return 0;
}
