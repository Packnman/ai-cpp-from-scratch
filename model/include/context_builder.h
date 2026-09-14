#pragma once

#include "tokenizer_conversation.h"
#include <cstddef>
#include <string>
#include <vector>

// 質問と回答を分離して保持し、選択時に発話途中で切断されない会話ターン。
struct ConversationContextTurn {
        std::string strQuestion;
        std::string strResponse;
};

// 構築済みtoken列と、採用された入力要素の元indexを返す診断可能な結果。
struct BuiltContext {
        TokenIds nTokens;
        std::vector<std::size_t> nTurnIndices;
        std::vector<std::size_t> nDocumentSentenceIndices;
};

// 現在質問を必ず完全な形で残し、完全な履歴ターンを予算内で選択する。
BuiltContext g_buildConversationContext(
    const TokenConversation &c_tokTokenizer,
    const std::vector<ConversationContextTurn> &c_trnHistory,
    const std::string &c_strQuestion, std::size_t nMaxTokens);

// UTF-8を検証し、文末記号を直前の文に残したまま文書を分割する。
std::vector<std::string>
g_splitDocumentSentences(const std::string &c_strDocument);

// 関連文書文、完全な履歴ターン、現在質問から予算内のpromptを構築する。
BuiltContext g_buildDocumentContext(
    const TokenConversation &c_tokTokenizer, const std::string &c_strDocument,
    const std::vector<ConversationContextTurn> &c_trnHistory,
    const std::string &c_strQuestion, std::size_t nMaxTokens);
