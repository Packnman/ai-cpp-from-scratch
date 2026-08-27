#include "matrix.h"
#include "model.h"
#include "optimizer.h"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
namespace {
void require(bool v, const char *m) {
  if (!v)
    throw std::runtime_error(m);
}
std::vector<float> hostmat(const cuMat &m) {
  Mat h(m._nRows, m._nCols);
  m.upload(h);
  return {h._lpfHost, h._lpfHost + m._nRows * m._nCols};
}
std::vector<float> host(Tensor *t) { return hostmat(t->_mData); }
void same(MnistModel &a, MnistModel &b) {
  auto x = a.getParams(), y = b.getParams();
  for (std::size_t i = 0; i < x.size(); ++i)
    require(host(x[i]) == host(y[i]), "parameter mismatch");
}
template <class F> void rejected(F f) {
  try {
    f();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error("expected rejection");
}
} // namespace
int main() {
  const std::string model_path = "/tmp/ai_cpp_model_test.bin",
                    ckpt_path = "/tmp/ai_cpp_checkpoint_test.bin",
                    bad_path = "/tmp/ai_cpp_bad_test.bin";
  MnistModel a(7), b(9);
  auto input = std::make_shared<Tensor>(784, 2),
       target = std::make_shared<Tensor>(10, 2);
  cuda_fill(input->_mData, 0.25f);
  cuda_fill(target->_mData, 0);
  Mat labels(10, 2);
  for (int i = 0; i < 20; ++i)
    labels._lpfHost[i] = 0;
  labels._lpfHost[0] = 1;
  labels._lpfHost[11] = 1;
  target->_mData.download(labels);
  std::vector<std::shared_ptr<Tensor>> args{input};
  auto logits = a.forward(args);
  require(logits->_mData._nRows == 10 && logits->_mData._nCols == 2,
          "forward shape");
  auto loss = a.loss(input, target);
  loss->backward();
  bool nonzero = false;
  for (auto *p : a.getParams())
    for (float v : hostmat(p->_mGrad))
      nonzero |= std::isfinite(v) && v != 0;
  require(nonzero, "backward/initialization");
  a.zero_grads();
  for (auto *p : a.getParams())
    for (float v : hostmat(p->_mGrad))
      require(v == 0, "zero grads");
  a.save(model_path.c_str());
  b.load(model_path.c_str());
  same(a, b);
  Adam oa(&a, 0.001f), ob(&b, 0.9f);
  rejected([&] { oa.save_checkpoint(ckpt_path.c_str()); });
  oa.init();
  for (auto *p : a.getParams())
    cuda_fill(p->_mGrad, 0.1f);
  oa.update();
  oa.save_checkpoint(ckpt_path.c_str());
  ob.load_checkpoint(ckpt_path.c_str());
  for (auto *p : a.getParams())
    cuda_fill(p->_mGrad, 0.2f);
  for (auto *p : b.getParams())
    cuda_fill(p->_mGrad, 0.2f);
  oa.update();
  ob.update();
  same(a, b);
  SGD wrong(&b, 0.1f);
  rejected([&] { wrong.load_checkpoint(ckpt_path.c_str()); });
  {
    FILE *file = std::fopen(bad_path.c_str(), "wb");
    if (!file)
      return 1;
    const char bad[] = "bad";
    if (std::fwrite(bad, 1, 3, file) != 3) {
      std::fclose(file);
      return 1;
    }
    std::fclose(file);
  }
  rejected([&] { b.load(bad_path.c_str()); });
  return 0;
}
