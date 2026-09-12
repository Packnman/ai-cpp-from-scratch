#include "conversation_runtime.h"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

using Json = nlohmann::json;

namespace
{
Json g_specials()
{
    return { { "pad", TokenConversation::PAD },
             { "unk", TokenConversation::UNK },
             { "begin", TokenConversation::BEGIN },
             { "end", TokenConversation::END },
             { "speaker_a", TokenConversation::SPEAKER_A },
             { "speaker_b", TokenConversation::SPEAKER_B },
             { "utterance_end", TokenConversation::UTTERANCE_END } };
}

} // namespace

void g_saveConversation( Transformer& trnModel, const TokenConversation& c_tokTokenizer,
                         const std::string& c_strDirectory )
{
    const auto& c_cfgModel = trnModel.config();
    if( c_cfgModel.nVocabulary != c_tokTokenizer.vocabSize() )
    {
        throw std::invalid_argument( "Model/tokenizer vocabulary mismatch" );
    }
    const std::filesystem::path pthDirectory( c_strDirectory );
    std::filesystem::create_directories( pthDirectory );
    // Write metadata last: a newly created bundle is readable only when complete.
    trnModel.save( ( pthDirectory / "weights.bin" ).c_str() );
    Json jsnManifest = { { "format", "ai_cpp_conversation" },
                         { "version", 1 },
                         { "dtype", "float32" },
                         { "vocabulary", c_tokTokenizer.vocabulary() },
                         { "special_ids", g_specials() },
                         { "config",
                           { { "vocabulary", c_cfgModel.nVocabulary },
                             { "blocks", c_cfgModel.nBlocks },
                             { "embedding", c_cfgModel.nEmbedding },
                             { "heads", c_cfgModel.nHeads },
                             { "hidden", c_cfgModel.nHidden },
                             { "context", c_cfgModel.nContext },
                             { "dropout", c_cfgModel.fDropout },
                             { "seed", c_cfgModel.nSeed } } } };
    std::ofstream ofsManifest( pthDirectory / "manifest.json" );
    ofsManifest << jsnManifest.dump( 2 ) << '\n';
    if( !ofsManifest )
    {
        throw std::runtime_error( "Cannot write model manifest" );
    }
}

ConversationBundle g_loadConversation( const std::string& c_strDirectory )
{
    const std::filesystem::path pthDirectory( c_strDirectory );
    std::ifstream ifsManifest( pthDirectory / "manifest.json" );
    Json jsnManifest;
    ifsManifest >> jsnManifest;
    if( jsnManifest.at( "format" ) != "ai_cpp_conversation" || jsnManifest.at( "version" ) != 1 ||
        jsnManifest.at( "dtype" ) != "float32" || jsnManifest.at( "special_ids" ) != g_specials() )
    {
        throw std::invalid_argument( "Unsupported conversation model format" );
    }
    const auto& c_jsnConfig = jsnManifest.at( "config" );
    TransformerConfig cfgModel;
    cfgModel.nVocabulary = c_jsnConfig.at( "vocabulary" ).get<int>();
    cfgModel.nBlocks = c_jsnConfig.at( "blocks" ).get<int>();
    cfgModel.nEmbedding = c_jsnConfig.at( "embedding" ).get<int>();
    cfgModel.nHeads = c_jsnConfig.at( "heads" ).get<int>();
    cfgModel.nHidden = c_jsnConfig.at( "hidden" ).get<int>();
    cfgModel.nContext = c_jsnConfig.at( "context" ).get<int>();
    cfgModel.fDropout = c_jsnConfig.at( "dropout" ).get<float>();
    cfgModel.nSeed = c_jsnConfig.at( "seed" ).get<std::uint64_t>();
    TokenConversation tokTokenizer( jsnManifest.at( "vocabulary" ).get<std::string>() );
    if( tokTokenizer.vocabSize() != cfgModel.nVocabulary ||
        tokTokenizer.vocabulary() != jsnManifest.at( "vocabulary" ).get<std::string>() )
    {
        throw std::invalid_argument( "Noncanonical or mismatched vocabulary" );
    }
    auto spModel = std::make_unique<Transformer>( cfgModel );
    spModel->load( ( pthDirectory / "weights.bin" ).c_str() );
    spModel->setTraining( false );
    //
    return { std::move( spModel ), std::move( tokTokenizer ) };
}

