#include "graph.h"
#include "serialization.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>

namespace {
struct FileCloser {
  void operator()(FILE *file) const { std::fclose(file); }
};
constexpr std::array<char, 8> magic{{'A', 'I', 'C', 'P', 'P', 'M', 'D', 'L'}};
constexpr std::uint32_t version = 1;
void path_ok(const char *p) {
  if (!p || !*p) {
    throw std::runtime_error("Model persistence: invalid path");
  }
}
} // namespace
void Model::save(const char *path) const {
  path_ok(path);
  std::unique_ptr<FILE, FileCloser> out(std::fopen(path, "wb"), FileCloser{});
  if (!out) {
    throw std::runtime_error("Model::save: failed to open file");
  }
  auto params = const_cast<Model *>(this)->getParams();
  serialization::write_exact(out.get(), magic.data(), magic.size());
  serialization::write(out.get(), version);
  serialization::write(out.get(), static_cast<std::uint32_t>(params.size()));
  for (auto *p : params) {
    if (!p) {
      throw std::runtime_error("Model::save: null parameter");
    }
    serialization::write_matrix(out.get(), p->_mData);
  }
}
void Model::load(const char *path) {
  path_ok(path);
  std::unique_ptr<FILE, FileCloser> in(std::fopen(path, "rb"), FileCloser{});
  if (!in) {
    throw std::runtime_error("Model::load: failed to open file");
  }
  std::array<char, 8> got{};
  serialization::read_exact(in.get(), got.data(), got.size());
  if (got != magic) {
    throw std::runtime_error("Model::load: invalid magic");
  }
  if (serialization::read<std::uint32_t>(in.get()) != version) {
    throw std::runtime_error("Model::load: unsupported version");
  }
  auto params = getParams();
  auto count = serialization::read<std::uint32_t>(in.get());
  if (count != params.size()) {
    throw std::runtime_error("Model::load: parameter count mismatch");
  }
  std::vector<serialization::MatrixData> data;
  data.reserve(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    if (!params[i]) {
      throw std::runtime_error("Model::load: null parameter");
    }
    data.push_back(serialization::read_matrix(in.get()));
    serialization::validate(data.back(), params[i]->_mData);
  }
  serialization::eof(in.get());
  for (std::uint32_t i = 0; i < count; ++i) {
    serialization::apply(data[i], params[i]->_mData);
  }
}
