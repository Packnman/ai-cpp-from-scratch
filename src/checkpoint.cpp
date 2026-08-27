#include "graph.h"
#include "optimizer.h"
#include "serialization.h"
#include "tensor.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
namespace {
struct FileCloser {
  void operator()(FILE *file) const { std::fclose(file); }
};
constexpr std::array<char, 8> magic{{'A', 'I', 'C', 'P', 'P', 'C', 'K', 'P'}};
constexpr std::uint32_t version = 1;
void valid(const char *p) {
  if (!p || !*p)
    throw std::runtime_error("checkpoint: invalid path");
}
} // namespace
void Optimizer::save_checkpoint(const char *path) const {
  if (!_initialized)
    throw std::runtime_error(
        "Optimizer::save_checkpoint: optimizer is not initialized");
  valid(path);
  std::unique_ptr<FILE, FileCloser> out(std::fopen(path, "wb"), FileCloser{});
  if (!out)
    throw std::runtime_error("Optimizer::save_checkpoint: open failed");
  serialization::write_exact(out.get(), magic.data(), magic.size());
  serialization::write(out.get(), version);
  serialization::write(out.get(), static_cast<std::uint32_t>(optimizerKind()));
  serialization::write(out.get(), _fLearningRate);
  serialization::write(out.get(), static_cast<std::int64_t>(_nStep));
  serialization::write(out.get(), static_cast<std::uint32_t>(_lpParams.size()));
  for (std::size_t i = 0; i < _lpParams.size(); ++i) {
    if (!_lpParams[i] || !_spOptimizerParams[i])
      throw std::runtime_error("Optimizer::save_checkpoint: invalid parameter");
    serialization::write_matrix(out.get(), _lpParams[i]->_mData);
    auto states = static_cast<const OptimizerParams &>(*_spOptimizerParams[i])
                      .stateMatrices();
    serialization::write(out.get(), static_cast<std::uint32_t>(states.size()));
    for (auto *s : states)
      serialization::write_matrix(out.get(), *s);
  }
}
void Optimizer::load_checkpoint(const char *path) {
  valid(path);
  std::unique_ptr<FILE, FileCloser> in(std::fopen(path, "rb"), FileCloser{});
  if (!in)
    throw std::runtime_error("Optimizer::load_checkpoint: open failed");
  std::array<char, 8> got{};
  serialization::read_exact(in.get(), got.data(), got.size());
  if (got != magic)
    throw std::runtime_error("Optimizer::load_checkpoint: invalid magic");
  if (serialization::read<std::uint32_t>(in.get()) != version)
    throw std::runtime_error("Optimizer::load_checkpoint: unsupported version");
  if (serialization::read<std::uint32_t>(in.get()) !=
      static_cast<std::uint32_t>(optimizerKind()))
    throw std::runtime_error(
        "Optimizer::load_checkpoint: optimizer type mismatch");
  float rate = serialization::read<float>(in.get());
  auto step = serialization::read<std::int64_t>(in.get());
  if (step < 0 || step > 2147483647)
    throw std::runtime_error("Optimizer::load_checkpoint: invalid step");
  auto params = _lpModel->getParams();
  auto count = serialization::read<std::uint32_t>(in.get());
  if (count != params.size())
    throw std::runtime_error(
        "Optimizer::load_checkpoint: parameter count mismatch");
  std::vector<serialization::MatrixData> values;
  std::vector<std::vector<serialization::MatrixData>> states(count);
  values.reserve(count);
  const std::uint32_t expected = optimizerKind() == 1 ? 2u : 0u;
  for (std::uint32_t i = 0; i < count; ++i) {
    if (!params[i])
      throw std::runtime_error("Optimizer::load_checkpoint: null parameter");
    values.push_back(serialization::read_matrix(in.get()));
    serialization::validate(values.back(), params[i]->_mData);
    auto n = serialization::read<std::uint32_t>(in.get());
    if (n != expected)
      throw std::runtime_error(
          "Optimizer::load_checkpoint: state count mismatch");
    for (std::uint32_t j = 0; j < n; ++j) {
      states[i].push_back(serialization::read_matrix(in.get()));
      serialization::validate(states[i].back(), params[i]->_mData);
    }
  }
  serialization::eof(in.get());
  init();
  for (std::uint32_t i = 0; i < count; ++i) {
    serialization::apply(values[i], _lpParams[i]->_mData);
    auto targets = _spOptimizerParams[i]->stateMatrices();
    for (std::size_t j = 0; j < targets.size(); ++j)
      serialization::apply(states[i][j], *targets[j]);
  }
  _fLearningRate = rate;
  _nStep = static_cast<int>(step);
}
