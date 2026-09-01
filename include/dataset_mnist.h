#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "trainer.h"

class MnistDataset {
public:
    static MnistDataset load(
        const std::string& c_strImagePath,
        const std::string& c_strLabelPath
    );

    std::size_t size() const;

    SupervisedBatch makeBatch(
        const std::vector<std::size_t>& c_nIndices, // シャッフルされた画像番号
        std::size_t nBegin,                         // indicesのどこから使うか
        std::size_t nCount                          // 今回何枚使うか
    ) const;

private:
    std::vector<std::uint8_t> _nImages;
    std::vector<std::uint8_t> _nLabels;
};
