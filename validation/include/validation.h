#pragma once

#include "conversation_runtime.h"
#include <ostream>

struct ConfigGeneration
{
    float fTemperature = 0.8f;
    int nTopK = 40;
    int nMaxTokens = 128;
};

TokenIds
Generate(
    Transformer& trnModel,
    const TokenConversation& c_tokTokenizer,
    const TokenIds& c_nHistory,
    std::mt19937& rngRandom,
    const ConfigGeneration& c_cfgGeneration = {}
);

// Evaluate a saved bundle without optimizer updates or model writes.
double
Validation(
    const std::string& c_strDataDirectory,
    const std::string& c_strModelDirectory,
    const std::string& c_strSplit, int nBatchSize,
    int nMaxBatches, std::ostream& stmLog
);
