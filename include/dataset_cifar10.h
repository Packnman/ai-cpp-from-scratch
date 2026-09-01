#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "trainer.h"

class Cifar10Dataset
{
public:
    static Cifar10Dataset loadTraining(const std::string& c_strDirectory);
    static Cifar10Dataset loadTest(const std::string& c_strDirectory);

    std::size_t size() const;

    SupervisedBatch makeBatch(
        const std::vector<std::size_t>& c_nIndices,
        std::size_t nBegin,
        std::size_t nCount
    ) const;

private:
    static Cifar10Dataset loadFiles(
        const std::string& c_strDirectory,
        const std::vector<std::string>& c_strFileNames
    );
    void appendFile(const std::string& c_strPath);

    std::vector<std::uint8_t> _nImages;
    std::vector<std::uint8_t> _nLabels;
};
