#pragma once

#include <cstddef>
#include <filesystem>

namespace ai::model {

void shard_jawiki(const std::filesystem::path &source,
                  const std::filesystem::path &output,
                  std::size_t target_bytes);

} // namespace ai::model
