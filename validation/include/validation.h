#pragma once

#include "context_builder.h"
#include "conversation_runtime.h"
#include <functional>
#include <istream>
#include <ostream>

struct ConfigGeneration
{
    float fTemperature = 0.8f;
    int nTopK = 40;
    int nMaxTokens = 256;
    int nContext = 0; // 0: saved model context; positive: input window <= saved context.
};

TokenIds
Generate(
    Transformer& trnModel,
    const TokenConversation& c_tokTokenizer,
    const TokenIds& c_nHistory,
    std::mt19937& rngRandom,
    const ConfigGeneration& c_cfgGeneration = {}
);

using ConversationGenerator = std::function<TokenIds( const TokenIds& )>;

void Chat( const TokenConversation& c_tokTokenizer, int nContext,
           std::istream& stmInput, std::ostream& stmOutput,
           const ConversationGenerator& c_fnGenerate );

void DocumentChat( const TokenConversation& c_tokTokenizer,
                   const std::string& c_strDocument, int nInputBudget,
                   std::istream& stmInput, std::ostream& stmOutput,
                   const ConversationGenerator& c_fnGenerate );

// Evaluate a saved bundle without optimizer updates or model writes.
double
Validation(
    const std::string& c_strDataDirectory,
    const std::string& c_strModelDirectory,
    const std::string& c_strSplit, int nBatchSize,
    int nMaxBatches, std::ostream& stmLog,
    ConversationLossTarget enmTarget = ConversationLossTarget::All
);
