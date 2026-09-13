#include "train.h"
#include "validation.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unistd.h>
namespace
{
void require( bool condition, const char* message )
{
    if ( !condition )
    {
        throw std::runtime_error( message );
    }
}
template <class F> void rejects( F call )
{
    bool failed = false;
    try
    {
        call();
    }
    catch ( const std::exception& )
    {
        failed = true;
    }
    require( failed, "Expected rejection" );
}
} // namespace
int main()
{
    const auto root =
        std::filesystem::temp_directory_path() / ( "ai-cpp-subword-check-" + std::to_string( getpid() ) );
    try
    {
        std::filesystem::create_directories( root );
        auto tokenizer = TokenConversation::trainSubword(
            { "こんにちは、物理の説明です。速度と時間。abc123", "日本語 Markdown $v=at$" }, 512 );
        const std::vector<std::string> examples = {
            "",
            " \t\r\n  ",
            "未知の漢字𠮷🧪",
            "ＡＢＣ é é １２３",
            "# 力学\n\n  - 速度 $v = at$  \n$$\n\\frac{1}{2}mv^2 = E\n$$\n",
            "~~~cpp\n\tint x = 2;\n~~~\n",
            "▁ ▁<unk><s><speaker_a><0x00>",
            std::string( "a\0b", 3 ) };
        for ( const auto& text : examples )
        {
            const auto ids = tokenizer.encode( text );
            require( tokenizer.decode( ids ) == text, "Subword roundtrip" );
            require( std::none_of( ids.begin(), ids.end(), []( int id ) { return id < 7; } ),
                     "Text injected special/UNK" );
        }
        rejects( [&] { tokenizer.encode( std::string( "\xc0\x80", 2 ) ); } );
        rejects( [&] { tokenizer.decode( { -1 } ); } );
        rejects( [&] { tokenizer.decode( { tokenizer.vocabSize() } ); } );
        rejects( [&] { TokenConversation::fromSubwordModel( "corrupt" ); } );
        auto restored = TokenConversation::fromSubwordModel( tokenizer.subwordModel() );
        for ( const auto& text : examples )
        {
            require( restored.encode( text ) == tokenizer.encode( text ), "Serialized IDs changed" );
        }
        std::cout << "Subword roundtrip/bytes/whitespace/serialization passed\n";
        TransformerConfig config;
        config.nVocabulary = tokenizer.vocabSize();
        config.nBlocks = 1;
        config.nEmbedding = 8;
        config.nHeads = 2;
        config.nHidden = 16;
        config.nContext = 512;
        config.fDropout = 0;
        Transformer model( config );
        g_saveConversation( model, tokenizer, ( root / "bundle" ).string() );
        auto loaded = g_loadConversation( ( root / "bundle" ).string() );
        require( loaded.tokTokenizer.subwordModel() == tokenizer.subwordModel(),
                 "Bundle tokenizer mismatch" );
        require( loaded.spModel->config().nContext == 512, "Context not saved" );
        const auto data = root / "data";
        std::filesystem::create_directories( data );
        std::ofstream( data / "train.jsonl" )
            << "{\"id\":1,\"utterances\":[{\"speaker\":0,\"text\":\"abc abc abc\"}]}\n";
        std::ofstream( data / "validation.jsonl" )
            << "{\"id\":2,\"utterances\":[{\"speaker\":0,\"text\":\"🧪未知\"}]}\n";
        std::ofstream( data / "test.jsonl" )
            << "{\"id\":3,\"utterances\":[{\"speaker\":1,\"text\":\"資料外\"}]}\n";
        ConfigTraining training;
        training.nEpochs = 1;
        training.nBatchSize = 1;
        training.nMaxBatches = 1;
        std::ostringstream log;
        Finetuning( data.string(), ( root / "bundle" ).string(), ( root / "finetuned" ).string(), training,
                    log );
        auto finetuned = g_loadConversation( ( root / "finetuned" ).string() );
        require( finetuned.tokTokenizer.subwordModel() == tokenizer.subwordModel(),
                 "Finetuning rebuilt tokenizer" );
        rejects( [&] { Training( data.string(), ( root / "bundle" ).string(), config, training, log ); } );
        training.strTokenizer = "bpe";
        training.nTokenizerVocabulary = 512;
        Training( data.string(), ( root / "new" ).string(), config, training, log );
        auto fresh = g_loadConversation( ( root / "new" ).string() );
        auto expected = TokenConversation::trainSubword( { "abc abc abc" }, 512 );
        require( fresh.tokTokenizer.subwordModel() == expected.subwordModel(),
                 "Evaluation data entered tokenizer" );
        for ( auto* parameter : model.getParams() )
        {
            cuda_fill( parameter->_mData, 0.0f );
        }
        const int chosen = tokenizer.encode( "\n" ).at( 0 );
        Tensor* bias = nullptr;
        for ( const auto& parameter : model.namedParameters() )
        {
            if ( parameter.strName == "output_bias" )
            {
                bias = parameter.lpTensor;
            }
        }
        require( bias != nullptr, "Missing output bias" );
        auto values = bias->_mData.toHost();
        values.at( chosen ) = 10.0f;
        bias->_mData.copyFromHost( values.data(), values.size() );
        std::mt19937 random( 42 );
        ConfigGeneration generation;
        generation.nTopK = 1;
        generation.nMaxTokens = 129;
        TokenIds history( 512, chosen );
        require( Generate( model, tokenizer, history, random, generation ) == TokenIds( 129, chosen ),
                 "512 input / 129 output" );
        generation.nContext = 128;
        generation.nMaxTokens = 1;
        require( Generate( model, tokenizer, history, random, generation ) == TokenIds( 1, chosen ),
                 "Input context override" );
        generation.nContext = 513;
        rejects( [&] { Generate( model, tokenizer, history, random, generation ); } );
        generation.nContext = 0;
        history[0] = tokenizer.vocabSize();
        rejects( [&] { Generate( model, tokenizer, history, random, generation ); } );
        generation.nContext = 128;
        require( Generate( model, tokenizer, history, random, generation ).size() == 1, "Input cropping" );
        std::filesystem::remove( root / "bundle/tokenizer.model" );
        rejects( [&] { g_loadConversation( ( root / "bundle" ).string() ); } );
        std::filesystem::remove_all( root );
        std::cout << "BPE training/bundle/finetuning/512-token generation passed\n";
        return 0;
    }
    catch ( const std::exception& error )
    {
        std::cerr << error.what() << "\nArtifacts: " << root << '\n';
        return 1;
    }
}
