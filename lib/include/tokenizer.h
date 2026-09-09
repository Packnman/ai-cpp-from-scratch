#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "cuda_matrix.h"

using TokenIds =std::vector<std::int32_t>;
using DeviceTokenIds =std::shared_ptr<cunMat>;

template<class InputView,class Decoded>
class Tokenizer
{
public:
    virtual ~Tokenizer() =default;

    virtual TokenIds encode(InputView input) const =0;
    virtual Decoded decode(const TokenIds& tokenIds) const =0;
    virtual std::size_t vocabSize() const noexcept =0;

    DeviceTokenIds encodeDevice(InputView input) const
    {
        const TokenIds tokenIds =encode( input );
        auto result =std::make_shared<cunMat>(
            std::vector<std::int64_t>{
                static_cast<std::int64_t>(tokenIds.size()),
                1
            }
        );
        result->copyFromHost( tokenIds.data(),tokenIds.size() );
        return result;
    }
};
