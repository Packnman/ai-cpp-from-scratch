#include "cli_options.h"
#include "train.h"
#include <fstream>
#include <iostream>

int main(int nArgc, char **lpArgv) {
    try {
        if (nArgc >= 2 && std::string(lpArgv[1]) == "tokenizer") {
            if (nArgc < 5)
                throw std::invalid_argument(
                    "Usage: main_train tokenizer CONVERSATION_TRAIN TEXT_TRAIN OUTPUT "
                    "[--vocab-size N]");
            CliOptions options(nArgc, lpArgv, 5);
            const int vocabulary = options.integer("--vocab-size", 4096);
            options.finish();
            std::vector<std::string> sentences;
            for (const auto &dialogue : g_readConversations(lpArgv[2]))
                for (const auto &utterance : dialogue.uttUtterances)
                    sentences.push_back(utterance.strText);
            for (const auto &document : g_readTextDocuments(lpArgv[3]))
                sentences.push_back(document.strText);
            const auto tokenizer =
                TokenConversation::trainSubword(sentences, vocabulary);
            std::ofstream output(lpArgv[4], std::ios::binary);
            const auto &bytes = tokenizer.subwordModel();
            output.write(bytes.data(), bytes.size());
            if (!output)
                throw std::runtime_error("Cannot write tokenizer model");
            std::cout << "Saved common training-only tokenizer vocabulary="
                      << tokenizer.vocabSize() << '\n';
            return 0;
        }
        // prepare は元データを train / validation / test に変換する専用の入口。
        if (nArgc >= 2 && std::string(lpArgv[1]) == "prepare") {
            if (nArgc < 4 || nArgc > 5)
                throw std::invalid_argument(
                    "Usage: main_train prepare SOURCE_REPO OUTPUT_DIR "
                    "[REVISION]");
            g_prepareConversations(lpArgv[2], lpArgv[3],
                                   nArgc == 5 ? lpArgv[4] : "unspecified");
            std::cout << "Prepared train/validation/test and metadata.json\n";
            return 0;
        }
        if (nArgc < 3)
            throw std::invalid_argument(
                "Usage: main_train DATA MODEL [--resume] [--from-model SOURCE] "
                "[--epochs N --batch N --lr F "
                "--clip F --seed N --dropout F --blocks N --embedding N "
                "--heads N "
                "--hidden N "
                "--context N --tokenizer character|bpe --vocab-size N "
                "--loss-target all|response "
                "--data-format conversation|text --tokenizer-model FILE "
                "--token-budget N --accumulate N "
                "--max-batches "
                "N] | main_train prepare SOURCE_REPO OUTPUT_DIR [REVISION]");
        // DATA と MODEL に続くオプションを、モデル設定と学習条件に振り分ける。
        CliOptions optOptions(nArgc, lpArgv, 3);
        const bool isResume = optOptions.flag("--resume");
        const bool isFinetuning = optOptions.contains("--from-model");
        if (isResume && isFinetuning)
            throw std::invalid_argument(
                "--resume and --from-model cannot be used together");
        if (isResume) {
            for (const auto *option :
                 {"--batch", "--clip", "--seed", "--max-batches", "--blocks",
                  "--embedding", "--heads", "--hidden", "--context",
                  "--dropout", "--tokenizer", "--vocab-size", "--data-format",
                  "--tokenizer-model", "--token-budget", "--accumulate"})
                if (optOptions.contains(option))
                    throw std::invalid_argument(
                        std::string("Cannot override option when resuming: ") +
                        option);
        }
        const int nSeed = isResume ? 42 : optOptions.integer("--seed", 42);
        if (nSeed < 0)
            throw std::invalid_argument("Seed must be nonnegative");
        const auto strSource = optOptions.string("--from-model");
        // 保存重みと形状が食い違わないよう、追加学習ではモデル構成の上書きを拒否する。
        if (isFinetuning) {
            for (const auto *lpOption :
                 {"--blocks", "--embedding", "--heads", "--hidden", "--context",
                  "--dropout", "--tokenizer", "--vocab-size", "--tokenizer-model"}) {
                if (optOptions.contains(lpOption))
                    throw std::invalid_argument(
                        std::string("Cannot override saved model option: ") +
                        lpOption);
            }
        }
        TransformerConfig cfgModel;
        {
            cfgModel.nBlocks = optOptions.integer("--blocks", cfgModel.nBlocks);
            cfgModel.nEmbedding =
                optOptions.integer("--embedding", cfgModel.nEmbedding);
            cfgModel.nHeads = optOptions.integer("--heads", cfgModel.nHeads);
            cfgModel.nHidden = optOptions.integer("--hidden", cfgModel.nHidden);
            cfgModel.nContext =
                optOptions.integer("--context", cfgModel.nContext);
            cfgModel.fDropout = optOptions.real("--dropout", cfgModel.fDropout);
        }
        ConfigTraining cfgTraining;
        const bool hasLearningRate = optOptions.contains("--lr");
        const bool hasLossTarget = optOptions.contains("--loss-target");
        {
            cfgTraining.strTokenizer =
                optOptions.string("--tokenizer", "character");
            cfgTraining.nTokenizerVocabulary =
                optOptions.integer("--vocab-size", 4096);
            cfgTraining.strLossTarget =
                optOptions.string("--loss-target", "all");
            cfgTraining.strDataFormat =
                optOptions.string("--data-format", "conversation");
            cfgTraining.strTokenizerModel =
                optOptions.string("--tokenizer-model");
            const int tokenBudget = optOptions.integer("--token-budget", 0);
            if (tokenBudget < 0)
                throw std::invalid_argument("Token budget must be nonnegative");
            cfgTraining.nTokenBudget = static_cast<std::uint64_t>(tokenBudget);
            cfgTraining.nAccumulationSteps =
                optOptions.integer("--accumulate", 1);
            cfgTraining.nSeed = nSeed;
            cfgTraining.nEpochs =
                optOptions.integer("--epochs", cfgTraining.nEpochs);
            cfgTraining.nBatchSize =
                optOptions.integer("--batch", cfgTraining.nBatchSize);
            cfgTraining.fLearningRate =
                optOptions.real("--lr", cfgTraining.fLearningRate);
            cfgTraining.fClipNorm =
                optOptions.real("--clip", cfgTraining.fClipNorm);
            cfgTraining.nMaxBatches =
                optOptions.integer("--max-batches", cfgTraining.nMaxBatches);
        }
        // 読み取られていないオプションを検出し、指定ミスを見逃さない。
        optOptions.finish();
        if (isResume) {
            ResumeTraining(lpArgv[1], lpArgv[2], cfgTraining.nEpochs,
                           hasLearningRate
                               ? std::optional<float>(cfgTraining.fLearningRate)
                               : std::nullopt,
                           hasLossTarget
                               ? std::optional<std::string>(cfgTraining.strLossTarget)
                               : std::nullopt,
                           std::cout);
        } else if (isFinetuning) {
            Finetuning(lpArgv[1], strSource, lpArgv[2], cfgTraining, std::cout);
        } else {
            Training(lpArgv[1], lpArgv[2], cfgModel, cfgTraining, std::cout);
        }
        return 0;
    }
    // CLI の最上位で例外を受け取り、エラーメッセージと失敗の終了コードを返す。
    catch (const std::exception &c_excError) {
        std::cerr << c_excError.what() << '\n';
        return 1;
    }
}
