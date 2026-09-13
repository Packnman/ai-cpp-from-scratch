#pragma once

#include "conversation_runtime.h"
#include <optional>
#include <ostream>

struct ConfigTraining {
  int nEpochs = 10;
  int nBatchSize = 64;
  float fLearningRate = 3.0e-4f;
  float fClipNorm = 1.0f;
  std::uint64_t nSeed = 42;
  int nMaxBatches = 0; // 0 means all batches; positive values are smoke runs.
  std::string strTokenizer =
      "character"; // New models only; finetuning uses saved tokenizer.
  int nTokenizerVocabulary = 4096;
};

double ClipGradients(Model &mdlModel, float fMaximum);

void Training(const std::string &c_strDataDirectory,
              const std::string &c_strModelDirectory,
              TransformerConfig cfgModel, const ConfigTraining &c_cfgTraining,
              std::ostream &stmLog);
// Restart Adam from saved weights; output must be new or empty.
void Finetuning(const std::string &c_strDataDirectory,
                const std::string &c_strSourceModelDirectory,
                const std::string &c_strOutputDirectory,
                const ConfigTraining &c_cfgTraining, std::ostream &stmLog);

// Continue the same run from its latest complete epoch checkpoint.
void ResumeTraining(const std::string &c_strDataDirectory,
                    const std::string &c_strModelDirectory,
                    int nAdditionalEpochs, std::optional<float> fLearningRate,
                    std::ostream &stmLog);
