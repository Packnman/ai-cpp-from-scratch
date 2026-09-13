#include "train.h"
#include "cuda_memory.h"
#include "optimizer.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>

using Json = nlohmann::json;

namespace {
// CUDA の失敗を例外に変換し、呼出元のエラー処理へ伝える。
void _CheckCuda(cudaError_t cudStatus) {
  if (cudStatus != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(cudStatus));
  }
}

// 空の split と会話 ID の重複を拒否し、学習・評価データの混入を防ぐ。
void _CheckSplits(const std::vector<Conversation> &c_cnvTrain,
                  const std::vector<Conversation> &c_cnvValidation,
                  const std::vector<Conversation> &c_cnvTest) {
  std::set<std::int64_t> nIds;
  for (const auto *c_lpSplit : {&c_cnvTrain, &c_cnvValidation, &c_cnvTest}) {
    if (c_lpSplit->empty()) {
      throw std::invalid_argument("All splits must be nonempty");
    }
    for (const auto &c_cnvDialogue : *c_lpSplit) {
      if (!nIds.insert(c_cnvDialogue.nId).second) {
        throw std::invalid_argument("Dialogue leakage across splits");
      }
    }
  }
}

std::string _Fingerprint(const std::filesystem::path &c_pthFile) {
  std::ifstream stream(c_pthFile, std::ios::binary);
  if (!stream)
    throw std::runtime_error("Cannot fingerprint file: " + c_pthFile.string());
  std::uint64_t hash = 1469598103934665603ULL;
  char buffer[65536];
  while (stream) {
    stream.read(buffer, sizeof(buffer));
    for (std::streamsize i = 0; i < stream.gcount(); ++i) {
      hash ^= static_cast<unsigned char>(buffer[i]);
      hash *= 1099511628211ULL;
    }
  }
  std::ostringstream text;
  text << std::hex << hash;
  return text.str();
}

Json _ModelConfig(const TransformerConfig &c) {
  return {{"vocabulary", c.nVocabulary}, {"blocks", c.nBlocks},
          {"embedding", c.nEmbedding},   {"heads", c.nHeads},
          {"hidden", c.nHidden},         {"context", c.nContext},
          {"dropout", c.fDropout},       {"seed", c.nSeed}};
}

Json _DataFingerprints(const std::string &c_strDirectory) {
  return {{"train", _Fingerprint(c_strDirectory + "/train.jsonl")},
          {"validation", _Fingerprint(c_strDirectory + "/validation.jsonl")},
          {"test", _Fingerprint(c_strDirectory + "/test.jsonl")}};
}

std::string _TokenizerFingerprint(const std::filesystem::path &c_pthDirectory) {
  std::string result = _Fingerprint(c_pthDirectory / "manifest.json");
  if (std::filesystem::exists(c_pthDirectory / "tokenizer.model"))
    result += ':' + _Fingerprint(c_pthDirectory / "tokenizer.model");
  return result;
}

Json _ReadJson(const std::filesystem::path &c_pthFile) {
  std::ifstream stream(c_pthFile);
  if (!stream)
    throw std::runtime_error("Cannot read checkpoint file: " +
                             c_pthFile.string());
  Json value;
  stream >> value;
  return value;
}
} // namespace

// 全パラメータの勾配を同じ倍率で縮小する。戻り値は縮小前の全体 L2 ノルム。
double ClipGradients(Model &mdlModel, float fMaximum) {
  if (!std::isfinite(fMaximum) || fMaximum <= 0.0f) {
    throw std::invalid_argument("Clip norm must be finite and positive");
  }
  cublasHandle_t cblHandle;
  if (cublasCreate(&cblHandle) != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("Cannot create gradient norm handle");
  }
  // 各勾配の L2 ノルムを GPU で求め、二乗和からモデル全体のノルムを求める。
  double dblSquared = 0.0;
  try {
    for (const auto *c_lpParameter : mdlModel.getParams()) {
      float fNorm = 0.0f;
      if (c_lpParameter->_mGrad.numel() > std::numeric_limits<int>::max() ||
          cublasSnrm2(cblHandle,
                      static_cast<int>(c_lpParameter->_mGrad.numel()),
                      c_lpParameter->_mGrad.data(), 1,
                      &fNorm) != CUBLAS_STATUS_SUCCESS ||
          !std::isfinite(fNorm)) {
        throw std::runtime_error("Invalid gradient norm");
      }
      dblSquared += static_cast<double>(fNorm) * fNorm;
    }
  }
  // ノルム計算が途中で失敗した場合も cuBLAS のハンドルを解放する。
  catch (...) {
    cublasDestroy(cblHandle);
    throw;
  }
  cublasDestroy(cblHandle);
  const double dblNorm = std::sqrt(dblSquared);
  // 上限を超えたときだけ maximum / norm を掛け、勾配全体の方向を保つ。
  if (dblNorm > fMaximum) {
    for (auto *lpParameter : mdlModel.getParams()) {
      cuda_scale(lpParameter->_mGrad, static_cast<float>(fMaximum / dblNorm));
    }
  }
  return dblNorm;
}

namespace {
// 新規学習と追加学習の共通本体。モデル準備、各 split
// の実行、最良重みの保存を行う。
void _Prop(
    const std::string &c_strDataDirectory,
    const std::string &c_strModelDirectory,
    TransformerConfig cfgModel,
    const ConfigTraining &c_cfgTraining,
    std::ostream &stmLog,
    const std::string &c_strSource = {},
    bool isResume = false,
    std::optional<float> fResumeLearningRate = std::nullopt
)
{
    ConfigTraining cfgTraining = c_cfgTraining;
    if (cfgTraining.nEpochs <= 0 || cfgTraining.nBatchSize <= 0 ||
        cfgTraining.nMaxBatches < 0 ||
        !std::isfinite(cfgTraining.fLearningRate) ||
        cfgTraining.fLearningRate <= 0.0f ||
        !std::isfinite(cfgTraining.fClipNorm) || cfgTraining.fClipNorm <= 0.0f ||
        (fResumeLearningRate && (!std::isfinite(*fResumeLearningRate) ||
                                *fResumeLearningRate <= 0.0f))) {
      throw std::invalid_argument("Invalid training configuration");
    }
    // 学習・検証・最終テストのデータを読み込み、分割の妥当性を確認する。
    const auto cnvTrain =
        g_readConversations(c_strDataDirectory + "/train.jsonl");
    const auto cnvValidation =
        g_readConversations(c_strDataDirectory + "/validation.jsonl");
    const auto cnvTest = g_readConversations(c_strDataDirectory + "/test.jsonl");
    _CheckSplits(cnvTrain, cnvValidation, cnvTest);
    const bool isFinetuning = !c_strSource.empty() && !isResume;
    const std::filesystem::path pthCheckpoint =
        std::filesystem::path(c_strModelDirectory) / "checkpoint";
    Json jsnResume;
    std::string strPreviousCheckpoint;
    if (isResume) {
      const Json latest = _ReadJson(pthCheckpoint / "latest.json");
      if (latest.at("format") != "ai_cpp_training_latest" ||
          latest.at("version") != 1)
        throw std::runtime_error("Unsupported latest checkpoint format");
      strPreviousCheckpoint = latest.at("checkpoint").get<std::string>();
      if (std::filesystem::path(strPreviousCheckpoint).filename() !=
          strPreviousCheckpoint)
        throw std::runtime_error("Invalid checkpoint path");
      jsnResume = _ReadJson(pthCheckpoint / strPreviousCheckpoint);
      if (jsnResume.at("format") != "ai_cpp_training_checkpoint" ||
          jsnResume.at("version") != 1)
        throw std::runtime_error("Unsupported training checkpoint format");
      const auto &training = jsnResume.at("training");
      cfgTraining.nBatchSize = training.at("batch_size").get<int>();
      cfgTraining.fClipNorm = training.at("clip_norm").get<float>();
      cfgTraining.nSeed = training.at("seed").get<std::uint64_t>();
      cfgTraining.nMaxBatches = training.at("max_batches").get<int>();
      cfgTraining.fLearningRate = jsnResume.at("learning_rate").get<float>();
      if (fResumeLearningRate)
        cfgTraining.fLearningRate = *fResumeLearningRate;
    }

    {
      // パスを正規化して同一ディレクトリへの上書きと既存出力の破壊を防ぐ。
      const auto pthOutput =
          std::filesystem::weakly_canonical(c_strModelDirectory);
      const auto pthSource = isFinetuning
                                ? std::filesystem::canonical(c_strSource)
                                : std::filesystem::path{};
      if (pthOutput == pthSource ||
          (!isResume && std::filesystem::exists(pthOutput) &&
          (!std::filesystem::is_directory(pthOutput) ||
            !std::filesystem::is_empty(pthOutput)))) {
        throw std::invalid_argument(
            "Training output must be a different new or empty directory");
      }
    }
    // 追加学習では重みと語彙を読み込む。新規学習の語彙は train
    // の文章だけから作る。
    auto bunModel = [&]() -> ConversationBundle {
      if (isFinetuning || isResume) {
        return g_loadConversation(isResume ? c_strModelDirectory : c_strSource);
      }
      if( (cfgTraining.strTokenizer!="character")&&
          (cfgTraining.strTokenizer!="bpe") )
      {
          throw std::invalid_argument("Tokenizer must be character or bpe");
      }
      auto tokTokenizer = [&]() {
          if (cfgTraining.strTokenizer == "character")
              return TokenConversation(g_trainingText(cnvTrain));
          std::vector<std::string> sentences;
          for (const auto &dialogue : cnvTrain)
              for (const auto &utterance : dialogue.uttUtterances)
                  sentences.push_back(utterance.strText);
          //
          return TokenConversation::trainSubword(
              sentences,
              cfgTraining.nTokenizerVocabulary
          );
      }();
      cfgModel.nVocabulary = tokTokenizer.vocabSize();
      cfgModel.nSeed = cfgTraining.nSeed;
      return {std::make_unique<Transformer>(cfgModel), std::move(tokTokenizer)};
    }();
    auto &trnModel = *bunModel.spModel;
    const auto &tokTokenizer = bunModel.tokTokenizer;
    cfgModel = trnModel.config();
    ConversationDataset datTrain(cnvTrain, tokTokenizer, cfgModel.nContext);
    ConversationDataset datValidation(cnvValidation, tokTokenizer, cfgModel.nContext);
    ConversationDataset datTest(cnvTest, tokTokenizer, cfgModel.nContext);
    // 追加学習でも Adam の移動平均・更新回数は引き継がず、新しく初期化する。
    Adam optAdam(&trnModel, cfgTraining.fLearningRate);
    optAdam.init();
    std::mt19937 rngRandom(
        static_cast<std::mt19937::result_type>(cfgTraining.nSeed));
    std::filesystem::create_directories(c_strModelDirectory);
    std::filesystem::create_directories(pthCheckpoint);
    int nCompletedEpoch = 0;
    int nBestEpoch = 0;
    double dblBest = std::numeric_limits<double>::infinity();
    if( isResume )
    {
        if( (jsnResume.at("model")!=_ModelConfig(cfgModel))||
            (jsnResume.at("data")!=_DataFingerprints(c_strDataDirectory))||
            (jsnResume.at("tokenizer_fingerprint")!=_TokenizerFingerprint(c_strModelDirectory)) )
        {
            throw std::runtime_error("Checkpoint model, tokenizer, or data mismatch");
        }
        const auto weightName = jsnResume.at("weights").get<std::string>();
        const auto adamName = jsnResume.at("adam").get<std::string>();
        if( (std::filesystem::path(weightName).filename()!=weightName)||
            (std::filesystem::path(adamName).filename()!=adamName)||
            (jsnResume.at("weights_fingerprint")!=_Fingerprint(pthCheckpoint/weightName))||
            (jsnResume.at("adam_fingerprint")!=_Fingerprint(pthCheckpoint/adamName)) )
        {
            throw std::runtime_error("Checkpoint is missing or corrupt");
        }
        trnModel.load((pthCheckpoint / weightName).c_str());
        optAdam.loadState((pthCheckpoint / adamName).string());
        if( optAdam.step() != jsnResume.at("adam_step").get<std::uint64_t>() ||
            optAdam.learningRate() != jsnResume.at("learning_rate").get<float>() )
        {
            throw std::runtime_error("Checkpoint Adam metadata mismatch");
        }
        if( fResumeLearningRate )
        {
            optAdam.setLearningRate(*fResumeLearningRate);
        }
        std::istringstream randomState( jsnResume.at("shuffle_state").get<std::string>() );
        randomState >> rngRandom;
        if( !randomState )
        {
            throw std::runtime_error("Invalid checkpoint shuffle state");
        }
        trnModel.setDropoutCounters( jsnResume.at("dropout_counters").get<std::vector<std::uint64_t>>() );
        nCompletedEpoch = jsnResume.at("completed_epoch").get<int>();
        nBestEpoch = jsnResume.at("best_epoch").get<int>();
        dblBest = jsnResume.at("best_validation_loss").get<double>();
        if( nCompletedEpoch < 0 ||
            nBestEpoch < 0 ||
            nBestEpoch > nCompletedEpoch ||
            !std::isfinite(dblBest) )
        {
            throw std::runtime_error("Invalid checkpoint epoch state");
        }
    }
    std::ofstream ofsMetrics(c_strModelDirectory + "/metrics.jsonl", isResume ? std::ios::app : std::ios::trunc);
    if( !ofsMetrics )
    {
        throw std::runtime_error("Cannot write metrics");
    }
    stmLog << "vocabulary=" << cfgModel.nVocabulary
        << " windows=" << datTrain.size()
        << " max_batches=" << cfgTraining.nMaxBatches << '\n';
    const auto memoryMode = cu_memory::statistics();
    // 再現に必要な設定と実際のメモリ確保方式を、開始ログに記録する。
    const Json jsnTraining = {
        {"cuda_allocator", memoryMode.pooled ? "pool" : "legacy"},
        {"tokenizer", tokTokenizer.isSubword() ? "bpe" : "character"},
        {"vocabulary", tokTokenizer.vocabSize()},
        {"event", "training_start"},
        {"source_model", c_strSource},
        {"optimizer", "Adam newly initialized"},
        {"epochs", cfgTraining.nEpochs},
        {"batch_size", cfgTraining.nBatchSize},
        {"learning_rate", cfgTraining.fLearningRate},
        {"clip_norm", cfgTraining.fClipNorm},
        {"shuffle_seed", cfgTraining.nSeed},
        {"dropout_seed", cfgModel.nSeed},
        {"max_batches", cfgTraining.nMaxBatches},
        {"blocks", cfgModel.nBlocks},
        {"embedding", cfgModel.nEmbedding},
        {"heads", cfgModel.nHeads},
        {"hidden", cfgModel.nHidden},
        {"context", cfgModel.nContext},
        {"dropout", cfgModel.fDropout}
    };
    const Json jsnStart = isResume ? Json{
        {"event", "resume_start"},
        {"start_epoch", nCompletedEpoch + 1},
        {"additional_epochs", cfgTraining.nEpochs},
        {"checkpoint", strPreviousCheckpoint},
        {"learning_rate", optAdam.learningRate()},
        {"learning_rate_changed",
            fResumeLearningRate.has_value() &&
            *fResumeLearningRate != jsnResume.at("learning_rate").get<float>()}
    } : jsnTraining;
    stmLog << jsnStart.dump() << std::endl;
    ofsMetrics << jsnStart.dump() << std::endl;
    // 1つの split を処理する。学習時だけ dropout と重み更新を有効にする。
    auto fnRun = [&](
        const ConversationDataset &c_datData,
        bool isTraining,
        const std::string &c_strSplit,
        int nEpoch
    )
    {
        trnModel.setTraining(isTraining);
        // 学習データだけ毎回並べ替える。評価は元の順序で処理する。
        std::vector<std::size_t> nOrder(c_datData.size());
        std::iota(nOrder.begin(), nOrder.end(), 0);
        if( isTraining )
        {
            std::shuffle(nOrder.begin(), nOrder.end(), rngRandom);
        }
        double dblLossSum = 0.0;
        std::size_t nValid = 0;
        std::size_t nPeakUsed = 0;
        std::size_t nBaselineFree = 0;
        std::size_t nTotal = 0;
        _CheckCuda(cudaMemGetInfo(&nBaselineFree, &nTotal));
        // 先行する GPU 処理を完了させてから、今回の split の計時を始める。
        _CheckCuda(cudaDeviceSynchronize());
        const auto clkStart = std::chrono::steady_clock::now();
        int nBatches = 0;
        for (std::size_t nStart = 0; nStart < nOrder.size();
            nStart += cfgTraining.nBatchSize)
        {
            if (cfgTraining.nMaxBatches > 0 && nBatches >= cfgTraining.nMaxBatches) {
                break;
            }
            // 入力 ID と次トークンの正解 ID をバッチ化し、GPU に転送する。
            const auto batBatch = c_datData.batch(nOrder, nStart, cfgTraining.nBatchSize);
            auto spmInputs = std::make_shared<cunMat>(batBatch.nSequence, batBatch.nBatch);
            cunMat mTargets(batBatch.nSequence, batBatch.nBatch);
            spmInputs->copyFromHost(batBatch.nInputs.data(), batBatch.nInputs.size());
            mTargets.copyFromHost(batBatch.nTargets.data(), batBatch.nTargets.size());
            // 前回の勾配を消して順伝播する。PAD を除いた平均損失で状態を確認する。
            trnModel.zero_grads();
            auto spmLoss = trnModel.loss(spmInputs, mTargets);
            const float fLoss = spmLoss->_mData.toHost()[0];
            if( !std::isfinite(fLoss) )
            {
                throw std::runtime_error("Nonfinite loss");
            }
            if( isTraining )
            {
                // 逆伝播 → 勾配の大きさ制限 → Adam の順で重みを更新する。
                spmLoss->backward();
                ClipGradients(trnModel, cfgTraining.fClipNorm);
                optAdam.update();
            }
            std::size_t nFree = 0;
            _CheckCuda(cudaMemGetInfo(&nFree, &nTotal));
            nPeakUsed = std::max(nPeakUsed, nTotal - nFree);
            // バッチごとの平均を有効トークン数で重み付けし、端数バッチも正しく集計する。
            dblLossSum += static_cast<double>(fLoss) * batBatch.nValid;
            nValid += batBatch.nValid;
            ++nBatches;
        }
        _CheckCuda(cudaDeviceSynchronize());
        // 直前の同期で GPU の完了を待ち、実際の処理時間と処理量を算出する。
        const double dblSeconds = std::chrono::duration<double>(
                                      std::chrono::steady_clock::now() - clkStart)
                                      .count();
        // プールの予約量は再利用待ち領域も含むため、使用中 bytes と分けて記録する。
        const auto memoryStats = cu_memory::statistics();
        const double dblLoss = dblLossSum / nValid;
        Json jsnMetric = {{"epoch", nEpoch},
                          {"split", c_strSplit},
                          {"loss", dblLoss},
                          {"perplexity", std::exp(dblLoss)},
                          {"valid_tokens", nValid},
                          {"batches", nBatches},
                          {"seconds", dblSeconds},
                          {"tokens_per_second", nValid / dblSeconds},
                          {"sampled_device_used_bytes", nPeakUsed},
                          {"pool_used_bytes", memoryStats.usedBytes},
                          {"pool_reserved_bytes", memoryStats.reservedBytes},
                          {"baseline_device_used_bytes", nTotal - nBaselineFree},
                          {"max_batches", cfgTraining.nMaxBatches}};
        stmLog << jsnMetric.dump() << std::endl;
        ofsMetrics << jsnMetric.dump() << std::endl;
        //
        return dblLoss;
    };
    if (isFinetuning) {
        // 追加学習前のモデルも最良候補にする。悪化した場合はこの重みを残す。
        dblBest = fnRun(datValidation, false, "validation", 0);
        g_saveConversation(trnModel, tokTokenizer, c_strModelDirectory);
    }
    auto fnCheckpoint = [&](int nEpoch) {
      const std::string stem = "epoch-" + std::to_string(nEpoch);
      const std::string weightName = stem + ".weights.bin";
      const std::string adamName = stem + ".adam.bin";
      const std::string metadataName = stem + ".json";
      trnModel.save((pthCheckpoint / weightName).c_str());
      optAdam.saveState((pthCheckpoint / adamName).string());
      std::ostringstream randomState;
      randomState << rngRandom;
      Json metadata = {
          {"format", "ai_cpp_training_checkpoint"},
          {"version", 1},
          {"completed_epoch", nEpoch},
          {"best_validation_loss", dblBest},
          {"best_epoch", nBestEpoch},
          {"learning_rate", optAdam.learningRate()},
          {"adam_step", optAdam.step()},
          {"weights", weightName},
          {"adam", adamName},
          {"weights_fingerprint", _Fingerprint(pthCheckpoint / weightName)},
          {"adam_fingerprint", _Fingerprint(pthCheckpoint / adamName)},
          {"shuffle_state", randomState.str()},
          {"dropout_counters", trnModel.dropoutCounters()},
          {"training", {
              {"batch_size", cfgTraining.nBatchSize},
              {"clip_norm", cfgTraining.fClipNorm},
              {"seed", cfgTraining.nSeed},
              {"max_batches", cfgTraining.nMaxBatches}}},
          {"data", _DataFingerprints(c_strDataDirectory)},
          {"model", _ModelConfig(cfgModel)},
          {"tokenizer_fingerprint", _TokenizerFingerprint(c_strModelDirectory)}};
      {
          std::ofstream output(pthCheckpoint / metadataName, std::ios::trunc);
          output << metadata.dump(2) << '\n';
          if( !output )
          {
              throw std::runtime_error("Cannot write checkpoint metadata");
          }
      }
      const auto temporary = pthCheckpoint / "latest.json.tmp";
      {
          std::ofstream output(temporary, std::ios::trunc);
          output << Json{
              {"format", "ai_cpp_training_latest"},
              {"version", 1},
              {"checkpoint", metadataName}
          }.dump(2) << '\n';
          if( !output )
          {
              throw std::runtime_error("Cannot write latest checkpoint");
          }
      }
      std::filesystem::rename(temporary, pthCheckpoint / "latest.json");
      if (!strPreviousCheckpoint.empty()) {
          const auto oldWeights = jsnResume.value("weights", std::string());
          const auto oldAdam = jsnResume.value("adam", std::string());
          if (!oldWeights.empty() && oldWeights != weightName)
              std::filesystem::remove(pthCheckpoint / oldWeights);
          if (!oldAdam.empty() && oldAdam != adamName)
              std::filesystem::remove(pthCheckpoint / oldAdam);
          if (strPreviousCheckpoint != metadataName)
              std::filesystem::remove(pthCheckpoint / strPreviousCheckpoint);
      }
      strPreviousCheckpoint = metadataName;
      jsnResume = std::move(metadata);
    };
    for(int nEpoch = nCompletedEpoch + 1;
        nEpoch <= nCompletedEpoch + cfgTraining.nEpochs;
        ++nEpoch)
    {
        // Make temporary-allocation history (including an earlier test pass)
        // irrelevant at epoch boundaries.
        cu_memory::trimUnused();
        fnRun(datTrain, true, "train", nEpoch);
        const double dblValidation = fnRun(datValidation, false, "validation", nEpoch);
        // 検証損失が改善した場合だけ保存し、学習損失だけでモデルを選ばない。
        if (dblValidation < dblBest)
        {
            dblBest = dblValidation;
            nBestEpoch = nEpoch;
            g_saveConversation(trnModel, tokTokenizer, c_strModelDirectory);
        }
        fnCheckpoint(nEpoch);
    }
    // 最後の epoch ではなく、検証で最良だった重みを読み直して test を評価する。
    trnModel.load((c_strModelDirectory + "/weights.bin").c_str());
    fnRun(datTest, false, "test_best", nCompletedEpoch + cfgTraining.nEpochs);
}

} // namespace

// 新規モデルの設定を共通処理へ渡し、最初から学習する。
void Training(
    const std::string &c_strDataDirectory,
    const std::string &c_strModelDirectory,
    TransformerConfig cfgModel,
    const ConfigTraining &c_cfgTraining,
    std::ostream &stmLog
)
{
    _Prop(
        c_strDataDirectory,
        c_strModelDirectory,
        cfgModel,
        c_cfgTraining,
        stmLog
    );
}

// 保存済みモデルを出発点にする。モデル構成は読み込んだ設定を使用する。
void Finetuning(
    const std::string &c_strDataDirectory,
    const std::string &c_strSourceModelDirectory,
    const std::string &c_strOutputDirectory,
    const ConfigTraining &c_cfgTraining,
    std::ostream &stmLog
)
{
    if (c_strSourceModelDirectory.empty())
    {
      throw std::invalid_argument("Source model directory must not be empty");
    }

    _Prop(
        c_strDataDirectory,
        c_strOutputDirectory,
        {},
        c_cfgTraining,
        stmLog,
        c_strSourceModelDirectory
    );
}

void ResumeTraining(
    const std::string &c_strDataDirectory,
    const std::string &c_strModelDirectory,
    int nAdditionalEpochs,
    std::optional<float> fLearningRate,
    std::ostream &stmLog
)
{
    ConfigTraining config;
    config.nEpochs = nAdditionalEpochs;
    _Prop(
        c_strDataDirectory,
        c_strModelDirectory,
        {},
        config,
        stmLog,
        c_strModelDirectory,
        true,
        fLearningRate
    );
}
