#include "dataset_conversation.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include <set>
#include <stdexcept>

using Json = nlohmann::json;

namespace
{
void g_writeJson( const std::filesystem::path& c_pthFile, const Json& c_jsnValue )
{
    std::ofstream ofsFile( c_pthFile );
    ofsFile << c_jsnValue.dump( 2 ) << '\n';
    if( !ofsFile )
    {
        throw std::runtime_error( "Cannot write " + c_pthFile.string() );
    }
}
} // namespace

std::vector<Conversation> g_readConversations( const std::string& c_strFile )
{
    std::ifstream ifsFile( c_strFile );
    if( !ifsFile )
    {
        throw std::runtime_error( "Cannot read " + c_strFile );
    }
    std::vector<Conversation> cnvConversations;
    std::set<std::int64_t> nIds;
    std::string strLine;
    while( std::getline( ifsFile, strLine ) )
    {
        const auto jsnDialogue = Json::parse( strLine );
        Conversation cnvDialogue{ jsnDialogue.at( "id" ).get<std::int64_t>(), {} };
        if( !nIds.insert( cnvDialogue.nId ).second )
        {
            throw std::invalid_argument( "Duplicate dialogue ID" );
        }
        for( const auto& c_jsnUtterance : jsnDialogue.at( "utterances" ) )
        {
            const int nSpeaker = c_jsnUtterance.at( "speaker" ).get<int>();
            const std::string strText = c_jsnUtterance.at( "text" ).get<std::string>();
            if( nSpeaker != 0 && nSpeaker != 1 )
            {
                throw std::invalid_argument( "Unknown speaker" );
            }
            TokenCharacter tokValidate( strText );
            cnvDialogue.uttUtterances.push_back( { nSpeaker, strText } );
        }
        if( cnvDialogue.uttUtterances.empty() )
        {
            throw std::invalid_argument( "Empty dialogue" );
        }
        cnvConversations.push_back( std::move( cnvDialogue ) );
    }
    return cnvConversations;
}

void g_prepareConversations( const std::string& c_strSourceDirectory,
                             const std::string& c_strOutputDirectory,
                             const std::string& c_strRevision )
{
    const std::filesystem::path pthSource( c_strSourceDirectory );
    const std::filesystem::path pthOutput( c_strOutputDirectory );
    std::vector<std::filesystem::path> pthFiles;
    for( const auto& c_dirEntry :
         std::filesystem::directory_iterator( pthSource / "real_persona_chat/dialogues" ) )
    {
        if( c_dirEntry.path().extension() == ".json" )
        {
            pthFiles.push_back( c_dirEntry.path() );
        }
    }
    std::sort( pthFiles.begin(), pthFiles.end() );
    std::vector<Json> jsnDialogues;
    Json jsnDuplicates = Json::array();
    std::set<std::int64_t> nIds;
    for( const auto& c_pthFile : pthFiles )
    {
        std::ifstream ifsFile( c_pthFile );
        Json jsnOriginal;
        ifsFile >> jsnOriginal;
        const auto nId = jsnOriginal.at( "dialogue_id" ).get<std::int64_t>();
        const auto strSpeakers = jsnOriginal.at( "interlocutors" ).get<std::vector<std::string>>();
        if( strSpeakers.size() != 2 || strSpeakers[0] == strSpeakers[1] )
        {
            throw std::invalid_argument(
                "Invalid dialogue ID or interlocutors: " + c_pthFile.string() +
                " speakers=" + jsnOriginal.at( "interlocutors" ).dump() );
        }
        Json jsnUtterances = Json::array();
        for( const auto& c_jsnUtterance : jsnOriginal.at( "utterances" ) )
        {
            const auto strSpeaker = c_jsnUtterance.at( "interlocutor_id" ).get<std::string>();
            if( strSpeaker != strSpeakers[0] && strSpeaker != strSpeakers[1] )
            {
                throw std::invalid_argument( "Utterance speaker is not a dialogue interlocutor" );
            }
            const auto strText = c_jsnUtterance.at( "text" ).get<std::string>();
            TokenCharacter tokValidate( strText );
            jsnUtterances.push_back(
                { { "speaker", strSpeaker == strSpeakers[0] ? 0 : 1 }, { "text", strText } } );
        }
        if( jsnUtterances.empty() )
        {
            throw std::invalid_argument( "Empty dialogue" );
        }
        Json jsnDialogue = { { "id", nId }, { "utterances", jsnUtterances } };
        if( !nIds.insert( nId ).second )
        {
            const auto itrExisting = std::find_if( jsnDialogues.begin(), jsnDialogues.end(),
                                                   [&]( const Json& c_jsnValue )
                                                   {
                                                       return c_jsnValue.at( "id" ) == nId;
                                                   } );
            if( *itrExisting != jsnDialogue )
            {
                throw std::invalid_argument( "Conflicting duplicate dialogue ID: " +
                                             std::to_string( nId ) );
            }
            jsnDuplicates.push_back( { { "id", nId }, { "file", c_pthFile.filename().string() } } );
            continue;
        }
        jsnDialogues.push_back( std::move( jsnDialogue ) );
    }
    if( jsnDialogues.size() < 20 )
    {
        throw std::invalid_argument( "At least 20 dialogues are required for 90/5/5 splitting" );
    }
    std::sort( jsnDialogues.begin(), jsnDialogues.end(),
               []( const Json& c_jsnA, const Json& c_jsnB )
               {
                   return c_jsnA.at( "id" ).get<std::int64_t>() <
                          c_jsnB.at( "id" ).get<std::int64_t>();
               } );
    // Portable Fisher-Yates with rejection sampling; split before windowing.
    std::mt19937 rngRandom( 42 );
    for( std::size_t nSize = jsnDialogues.size(); nSize > 1; --nSize )
    {
        const std::uint64_t nRange = std::uint64_t{ 1 } << 32;
        const std::uint64_t nLimit = nRange - nRange % nSize;
        std::uint64_t nDraw;
        do
        {
            nDraw = rngRandom();
        } while( nDraw >= nLimit );
        std::swap( jsnDialogues[nSize - 1], jsnDialogues[nDraw % nSize] );
    }
    std::filesystem::create_directories( pthOutput );
    const std::size_t nTrain = jsnDialogues.size() * 90 / 100;
    const std::size_t nValidation = jsnDialogues.size() * 5 / 100;
    const std::vector<std::size_t> nBoundaries = { 0, nTrain, nTrain + nValidation,
                                                   jsnDialogues.size() };
    const std::vector<std::string> strSplits = { "train", "validation", "test" };
    Json jsnSplitIds;
    for( int nSplit = 0; nSplit < 3; ++nSplit )
    {
        std::ofstream ofsFile( pthOutput / ( strSplits[nSplit] + ".jsonl" ) );
        jsnSplitIds[strSplits[nSplit]] = Json::array();
        for( std::size_t nIndex = nBoundaries[nSplit]; nIndex < nBoundaries[nSplit + 1]; ++nIndex )
        {
            ofsFile << jsnDialogues[nIndex].dump() << '\n';
            jsnSplitIds[strSplits[nSplit]].push_back( jsnDialogues[nIndex].at( "id" ) );
        }
        if( !ofsFile )
        {
            throw std::runtime_error( "Cannot write split" );
        }
    }
    std::ifstream ifsVersion( pthSource / "VERSION" );
    if( !ifsVersion )
    {
        ifsVersion.open( pthSource / "real_persona_chat/VERSION" );
    }
    std::string strVersion;
    if( !std::getline( ifsVersion, strVersion ) )
    {
        throw std::runtime_error( "Missing corpus VERSION" );
    }
    g_writeJson(
        pthOutput / "metadata.json",
        { { "format", "ai_cpp_conversations_v1" },
          { "source", "https://github.com/nu-dialogue/real-persona-chat" },
          { "version", strVersion },
          { "revision", c_strRevision },
          { "seed", 42 },
          { "split_algorithm", "sorted ID, mt19937 rejection Fisher-Yates, floor 90/5/remainder" },
          { "dialogue_ids", jsnSplitIds },
          { "deduplicated_files", jsnDuplicates } } );
}

std::string g_trainingText( const std::vector<Conversation>& c_cnvConversations )
{
    std::string strText;
    for( const auto& c_cnvDialogue : c_cnvConversations )
    {
        for( const auto& c_uttUtterance : c_cnvDialogue.uttUtterances )
        {
            strText += c_uttUtterance.strText;
        }
    }
    return strText;
}

ConversationDataset::ConversationDataset( const std::vector<Conversation>& c_cnvConversations,
                                          const TokenConversation& c_tokTokenizer, int nContext )
    : _nContext( nContext )
{
    if( nContext <= 0 )
    {
        throw std::invalid_argument( "Context must be positive" );
    }
    for( const auto& c_cnvDialogue : c_cnvConversations )
    {
        TokenIds nIds = { TokenConversation::BEGIN };
        for( const auto& c_uttUtterance : c_cnvDialogue.uttUtterances )
        {
            if( c_uttUtterance.nSpeaker != 0 && c_uttUtterance.nSpeaker != 1 )
            {
                throw std::invalid_argument( "Invalid speaker" );
            }
            nIds.push_back( c_uttUtterance.nSpeaker == 0 ? TokenConversation::SPEAKER_A
                                                         : TokenConversation::SPEAKER_B );
            const auto nText = c_tokTokenizer.encode( c_uttUtterance.strText );
            nIds.insert( nIds.end(), nText.begin(), nText.end() );
            nIds.push_back( TokenConversation::UTTERANCE_END );
        }
        nIds.push_back( TokenConversation::END );
        for( std::size_t nStart = 0; nStart + 1 < nIds.size(); nStart += nContext )
        {
            const auto nEnd = std::min( nIds.size(), nStart + nContext + 1 );
            _nWindows.emplace_back( nIds.begin() + nStart, nIds.begin() + nEnd );
        }
    }
}

std::size_t ConversationDataset::size() const
{
    return _nWindows.size();
}

ConversationBatch ConversationDataset::batch( const std::vector<std::size_t>& c_nOrder,
                                              std::size_t nStart, int nBatchSize ) const
{
    if( nBatchSize <= 0 || nStart >= c_nOrder.size() )
    {
        throw std::invalid_argument( "Invalid batch range" );
    }
    const int nBatch = static_cast<int>(
        std::min( static_cast<std::size_t>( nBatchSize ), c_nOrder.size() - nStart ) );
    ConversationBatch batResult{
        _nContext, nBatch, 0,
        TokenIds( static_cast<std::size_t>( _nContext ) * nBatch, TokenConversation::PAD ),
        TokenIds( static_cast<std::size_t>( _nContext ) * nBatch, TokenConversation::PAD ) };
    for( int nColumn = 0; nColumn < nBatch; ++nColumn )
    {
        const auto& c_nWindow = _nWindows.at( c_nOrder[nStart + nColumn] );
        for( std::size_t nToken = 0; nToken < c_nWindow.size() &&
             nToken < static_cast<std::size_t>( _nContext ); ++nToken )
        {
            batResult.nInputs[nToken * nBatch + nColumn] = c_nWindow[nToken];
            if( nToken + 1 < c_nWindow.size() )
            {
                batResult.nTargets[nToken * nBatch + nColumn] = c_nWindow[nToken + 1];
                ++batResult.nValid;
            }
        }
    }
    return batResult;
}
