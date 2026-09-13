#pragma once

#include "tokenizer_conversation.h"
#include <cstdint>
#include <string>
#include <vector>

struct ConversationUtterance
{
    int nSpeaker;
    std::string strText;
};

struct Conversation
{
    std::int64_t nId;
    std::vector<ConversationUtterance> uttUtterances;
};

struct TextDocument
{
    std::int64_t nId;
    std::string strText;
};

struct ConversationBatch
{
    int nSequence;
    int nBatch;
    std::size_t nValid;
    TokenIds nInputs;
    TokenIds nTargets;
};

enum class ConversationLossTarget
{
    All,
    Response
};

ConversationLossTarget g_parseConversationLossTarget( const std::string& c_strValue );
const char* g_conversationLossTargetName( ConversationLossTarget enmTarget );

std::vector<Conversation> g_readConversations( const std::string& c_strFile );
void g_prepareConversations( const std::string& c_strSourceDirectory,
                             const std::string& c_strOutputDirectory,
                             const std::string& c_strRevision = "unspecified" );
std::string g_trainingText( const std::vector<Conversation>& c_cnvConversations );
std::vector<TextDocument> g_readTextDocuments( const std::string& c_strFile );

class ConversationDataset
{
public:
    ConversationDataset( const std::vector<Conversation>& c_cnvConversations,
                         const TokenConversation& c_tokTokenizer, int nContext = 128,
                         ConversationLossTarget enmTarget = ConversationLossTarget::All );
    ConversationDataset( const std::vector<TextDocument>& c_txtDocuments,
                         const TokenConversation& c_tokTokenizer, int nContext = 128 );
    std::size_t size() const;
    std::size_t excludedResponses() const;
    ConversationBatch batch( const std::vector<std::size_t>& c_nOrder, std::size_t nStart,
                             int nBatchSize ) const;

private:
    int _nContext;
    std::vector<TokenIds> _nWindows;
    std::vector<TokenIds> _nTargetWindows;
    std::size_t _nExcludedResponses = 0;
};
