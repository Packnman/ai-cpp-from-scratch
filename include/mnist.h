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
        const std::string& imagePath,
        const std::string& labelPath
    );

    std::size_t size() const;

    SupervisedBatch makeBatch(
        const std::vector<std::size_t>& indices,    // シャッフルされた画像番号
        std::size_t begin,                          // indicesのどこから使うか
        std::size_t count                           // 今回何枚使うか
    ) const;

private:
    std::vector<std::uint8_t> _images;
    std::vector<std::uint8_t> _labels;
};

std::size_t countClassificationCorrect(
    const std::shared_ptr<Tensor>& output,
    const std::shared_ptr<Tensor>& target
);
