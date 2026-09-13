#include "train.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

namespace
{
void require( bool condition, const char* message )
{
    if( !condition ) throw std::runtime_error( message );
}

std::string read( const std::filesystem::path& path )
{
    std::ifstream input( path, std::ios::binary );
    return std::string( std::istreambuf_iterator<char>( input ), {} );
}
}

int main()
{
    const auto root = std::filesystem::temp_directory_path() /
        ( "ai-cpp-text-pretraining-" + std::to_string( getpid() ) );
    try
    {
        const auto dataDirectory = root / "text";
        std::filesystem::create_directories( dataDirectory );
        std::ofstream( dataDirectory / "train.jsonl" )
            << "{\"id\":1,\"text\":\"日本語本文です。日本語本文です。\"}\n"
            << "{\"id\":2,\"text\":\"質問と回答答を学びます。\"}\n";
        std::ofstream( dataDirectory / "validation.jsonl" )
            << "{\"id\":3,\"text\":\"日本語本文です。\"}\n";
        std::ofstream( dataDirectory / "test.jsonl" )
            << "{\"id\":4,\"text\":\"質問と答です。\"}\n";

        const auto tokenizer = TokenConversation::trainSubword(
            { "日本語本文です。日本語本文です。", "質問と答を学びます。",
              "会話の質問です。", "会話の回答です。" }, 320 );
        const auto tokenizerPath = root / "common-tokenizer.model";
        {
            std::ofstream output( tokenizerPath, std::ios::binary );
            const auto& bytes = tokenizer.subwordModel();
            output.write( bytes.data(), bytes.size() );
        }

        const auto documents =
            g_readTextDocuments( ( dataDirectory / "train.jsonl" ).string() );
        ConversationDataset dataset( documents, tokenizer, 8 );
        const auto batch = dataset.batch( { 0 }, 0, 1 );
        require( batch.nInputs[0] == TokenConversation::BEGIN &&
                     batch.nTargets[0] != TokenConversation::PAD &&
                     batch.nValid > 0,
                 "Text dataset must use ordinary next-token targets" );

        TransformerConfig model;
        model.nBlocks = 1;
        model.nEmbedding = 16;
        model.nHeads = 2;
        model.nHidden = 32;
        model.nContext = 8;
        model.fDropout = 0.0f;
        ConfigTraining training;
        training.nEpochs = 3;
        training.nBatchSize = 1;
        training.fLearningRate = 0.01f;
        training.strDataFormat = "text";
        training.strTokenizer = "bpe";
        training.strTokenizerModel = tokenizerPath.string();
        training.nTokenBudget = 5;
        training.nAccumulationSteps = 2;
        std::ostringstream log;
        const auto modelDirectory = root / "model";
        Training( dataDirectory.string(), modelDirectory.string(), model, training, log );

        auto loaded = g_loadConversation( modelDirectory.string() );
        require( loaded.tokTokenizer.subwordModel() == tokenizer.subwordModel(),
                 "External common tokenizer must be saved unchanged" );
        std::ifstream metrics( modelDirectory / "metrics.jsonl" );
        std::string line;
        bool sawStart = false;
        bool sawTrain = false;
        while( std::getline( metrics, line ) )
        {
            const auto value = nlohmann::json::parse( line );
            if( value.value( "event", "" ) == "training_start" )
            {
                sawStart = value.at( "data_format" ) == "text" &&
                           value.at( "token_budget" ) == 5 &&
                           value.at( "accumulation_steps" ) == 2 &&
                           value.at( "effective_batch_tokens" ) == 16;
            }
            if( value.value( "split", "" ) == "train" )
            {
                sawTrain = value.at( "optimized_tokens" ).get<std::uint64_t>() >= 5;
            }
        }
        require( sawStart && sawTrain,
                 "Metrics must record text mode, accumulation and actual optimized tokens" );
        const auto latest = nlohmann::json::parse(
            read( modelDirectory / "checkpoint/latest.json" ) );
        const auto checkpoint = nlohmann::json::parse(
            read( modelDirectory / "checkpoint" /
                  latest.at( "checkpoint" ).get<std::string>() ) );
        require( checkpoint.at( "training" ).at( "data_format" ) == "text" &&
                     checkpoint.at( "training" ).at( "accumulation_steps" ) == 2 &&
                     checkpoint.at( "optimized_tokens" ).get<std::uint64_t>() >= 5,
                 "Checkpoint must preserve text/token-budget training state" );

        std::filesystem::remove_all( root );
        return 0;
    }
    catch( const std::exception& error )
    {
        std::cerr << error.what() << "\nArtifacts: " << root << '\n';
        return 1;
    }
}
