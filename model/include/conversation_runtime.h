#pragma once

#include "dataset_conversation.h"
#include "model_transformer.h"

struct ConversationBundle
{
    std::unique_ptr<Transformer> spModel;
    TokenConversation tokTokenizer;
};

void
g_saveConversation(
    Transformer& trnModel,
    const TokenConversation& c_tokTokenizer,
    const std::string& c_strDirectory
);
ConversationBundle
g_loadConversation(
    const std::string& c_strDirectory
);
