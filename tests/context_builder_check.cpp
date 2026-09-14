#include "context_builder.h"
#include "dataset_conversation.h"
#include <algorithm>
#include <random>
#include <stdexcept>

namespace {
void Require(bool isCondition, const char *lpMessage) {
    if (!isCondition)
        throw std::runtime_error(lpMessage);
}

bool Contains(const std::vector<std::size_t> &c_nValues, std::size_t nValue) {
    return std::find(c_nValues.begin(), c_nValues.end(), nValue) !=
           c_nValues.end();
}
} // namespace

int main() {
    // 会話context: token上限、決定性、最新60%とBM25関連40%の選択を検証する。
    const std::string vocabulary =
        "alpha project code 777 remembered unrelated newest answer question"
        "第一文。数値12345は針です！第三文?\n末尾";
    TokenConversation tokenizer(vocabulary);
    const std::vector<ConversationContextTurn> history = {
        {"alpha project 777", "remembered answer"},
        {"unrelated question", "unrelated answer"},
        {"newest question", "newest answer"}};
    const std::string question = "alpha 777";
    const auto base = g_buildConversationContext(tokenizer, {}, question, 1000);
    const auto oldTokens =
        g_buildConversationContext(tokenizer, {history[0]}, question, 1000)
            .nTokens.size() -
        base.nTokens.size();
    const auto newestTokens =
        g_buildConversationContext(tokenizer, {history[2]}, question, 1000)
            .nTokens.size() -
        base.nTokens.size();
    const std::size_t budget = base.nTokens.size() + oldTokens + newestTokens;
    const auto first =
        g_buildConversationContext(tokenizer, history, question, budget);
    const auto second =
        g_buildConversationContext(tokenizer, history, question, budget);
    Require(first.nTokens == second.nTokens &&
                first.nTurnIndices == second.nTurnIndices,
            "Context construction must be deterministic");
    Require(first.nTokens.size() <= budget,
            "Conversation token budget exceeded");
    Require(Contains(first.nTurnIndices, 2),
            "Newest complete turn was not retained");
    Require(Contains(first.nTurnIndices, 0),
            "BM25 did not retain the related old turn");
    Require(!Contains(first.nTurnIndices, 1),
            "Irrelevant turn displaced relevant context");

    // 重複除去: 完全一致するQAは最新の出現だけを保持する。
    const std::vector<ConversationContextTurn> duplicates = {
        history[0], history[0], history[2]};
    const auto deduplicated =
        g_buildConversationContext(tokenizer, duplicates, question, 1000);
    Require(!Contains(deduplicated.nTurnIndices, 0) &&
                Contains(deduplicated.nTurnIndices, 1),
            "Exact duplicate removal must retain the newest occurrence");

    bool isOverBudget = false;
    try {
        (void)g_buildConversationContext(tokenizer, {}, question, 2);
    } catch (const std::invalid_argument &) {
        isOverBudget = true;
    }
    Require(isOverBudget,
            "An indivisible over-budget question must be rejected");

    // 文書context: UTF-8の文境界保持と数値needleを含む文の選択を検証する。
    const std::string document = "第一文。数値12345は針です！第三文?\n末尾";
    const auto sentences = g_splitDocumentSentences(document);
    Require(
        sentences.size() == 5 && sentences[0] == "第一文。" &&
            sentences[1] == "数値12345は針です！" && sentences[3] == "\n",
        "Japanese, ASCII, and newline sentence boundaries must be preserved");
    const auto documentContext =
        g_buildDocumentContext(tokenizer, document, {history[2]}, "12345", 80);
    Require(
        documentContext.nTokens.size() <= 80 &&
            Contains(documentContext.nDocumentSentenceIndices, 1),
        "Document BM25 must select the sentence containing the numeric needle");

    // dataset: length bucketのseed再現性と動的paddingの集計を検証する。
    ConversationDataset dataset(
        std::vector<TextDocument>{{1, "a"},
                                  {2, "alpha project code"},
                                  {3, "alpha"},
                                  {4, "alpha project"}},
        tokenizer, 32);
    std::mt19937 randomA(91);
    std::mt19937 randomB(91);
    const auto orderA = dataset.lengthBucketedOrder(2, &randomA);
    const auto orderB = dataset.lengthBucketedOrder(2, &randomB);
    Require(orderA == orderB,
            "Length-bucket shuffle must be seed reproducible");
    const auto batch = dataset.batch({0, 1}, 0, 2);
    Require(batch.nSequence == static_cast<int>(dataset.sequenceLength(1)) &&
                batch.nTokens + batch.nPadding ==
                    static_cast<std::size_t>(batch.nSequence * batch.nBatch) &&
                batch.nPadding > 0,
            "Dynamic padding metrics are inconsistent");
    return 0;
}
