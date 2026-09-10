#include "tokenizer_character.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

static_assert(
    std::is_base_of_v<
        Tokenizer<std::string_view,std::string>,
        TokenCharacter
    >
);

namespace {
struct CardView
{
    std::int32_t rank;
    std::int32_t suit;
};

struct Card
{
    std::int32_t rank;
    std::int32_t suit;

    bool operator==(const Card&) const =default;
};

class CardTokenizer final : public Tokenizer<CardView,Card>
{
public:
    TokenIds encode(CardView card) const override
    {
        return {card.rank,card.suit+13};
    }

    Card decode(const TokenIds& tokenIds) const override
    {
        if( tokenIds.size()!=2 )
        {
            throw std::invalid_argument("CardTokenizer: expected two tokens");
        }
        return {tokenIds[0],tokenIds[1]-13};
    }

    std::size_t vocabSize() const noexcept override
    {
        return 17;
    }
};

void require(bool condition,const char* message)
{
    if( !condition ) throw std::runtime_error(message);
}

template<class Exception,class Callable>
void requireThrows(Callable&& callable,const char* message)
{
    try
    {
        callable();
    }
    catch( const Exception& )
    {
        return;
    }
    throw std::runtime_error(message);
}

void vocabularyAndRoundTrip()
{
    const std::string corpus ="To be,\nor not! \xc3\xa9\xce\xa9T";
    TokenCharacter tokenizer( corpus );

    const std::string codePointOrder ="\n !,Tbenort\xc3\xa9\xce\xa9";
    const std::vector<std::int32_t> expected{
        0,1,2,3,4,5,6,7,8,9,10,11,12
    };
    require(tokenizer.vocabSize()==expected.size(),"vocabulary size mismatch");
    require(
        tokenizer.encode(codePointOrder)==expected,
        "vocabulary is not ordered by Unicode code point"
    );
    require(
        tokenizer.decode(tokenizer.encode(corpus))==corpus,
        "encode/decode round trip failed"
    );

    auto device =tokenizer.encodeDevice("To\n");
    require(
        device->shape()==std::vector<std::int64_t>({3,1}),
        "encodeDevice shape mismatch"
    );
    require(
        device->toHost()==tokenizer.encode("To\n"),
        "encodeDevice values mismatch"
    );
}

void baseInterface()
{
    TokenCharacter characterTokenizer("abc");
    const Tokenizer<std::string_view,std::string>& tokenizer =
        characterTokenizer;

    require(tokenizer.encode("cab")==TokenIds({2,0,1}),"base encode dispatch failed");
    require(tokenizer.decode({2,0,1})=="cab","base decode dispatch failed");
    require(tokenizer.vocabSize()==3,"base vocabSize dispatch failed");

    CardTokenizer cardTokenizer;
    const Tokenizer<CardView,Card>& cardBase =cardTokenizer;
    const CardView queenOfHearts{11,2};
    const TokenIds expected{11,15};
    require(cardBase.encode(queenOfHearts)==expected,"structured encode failed");
    require(
        cardBase.decode(expected)==Card{11,2},
        "structured decode failed"
    );
    require(cardBase.vocabSize()==17,"structured vocabulary size mismatch");

    auto device =cardBase.encodeDevice( queenOfHearts );
    require(
        device->shape()==std::vector<std::int64_t>({2,1}),
        "generic encodeDevice shape mismatch"
    );
    require(device->toHost()==expected,"generic encodeDevice values mismatch");
}

void validation()
{
    TokenCharacter tokenizer("abc");

    requireThrows<std::out_of_range>(
        [&] { (void)tokenizer.encode("d"); },
        "unknown character was accepted"
    );
    requireThrows<std::out_of_range>(
        [&] { (void)tokenizer.decode({-1}); },
        "negative token ID was accepted"
    );
    requireThrows<std::out_of_range>(
        [&] { (void)tokenizer.decode({3}); },
        "out-of-range token ID was accepted"
    );
    requireThrows<std::invalid_argument>(
        [] { TokenCharacter invalid(std::string("\xc0\xaf",2)); },
        "overlong UTF-8 was accepted"
    );
    requireThrows<std::invalid_argument>(
        [&] { (void)tokenizer.encode(std::string("\xe2\x82",2)); },
        "truncated UTF-8 was accepted"
    );
    requireThrows<std::invalid_argument>(
        [] { TokenCharacter invalid(std::string("\xed\xa0\x80",3)); },
        "UTF-8 surrogate was accepted"
    );
    requireThrows<std::invalid_argument>(
        [] { TokenCharacter invalid(std::string("\xf4\x90\x80\x80",4)); },
        "out-of-range Unicode code point was accepted"
    );
}
} // namespace

int main()
{
    try
    {
        vocabularyAndRoundTrip();
        baseInterface();
        validation();
        std::cout<<"character tokenizer check passed\n";
        return 0;
    }
    catch( const std::exception& error )
    {
        std::cerr<<error.what()<<'\n';
        return 1;
    }
}
