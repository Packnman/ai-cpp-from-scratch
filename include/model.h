#pragma once
#include <cstdint>
#include "graph.h"

class MnistModel final : public Model {
public:
    explicit MnistModel(std::uint32_t seed=5489u);
    ~MnistModel();
private:
    std::shared_ptr<Tensor> _w1,_b1,_w2,_b2,_w3,_b3;
    Linear _linear1,_linear2,_linear3;
    ReLU _relu1,_relu2;
    SoftmaxCrossEntropy _criterion;
public:
    std::shared_ptr<Tensor> forward(
        std::vector<std::shared_ptr<Tensor>>& inputs
    ) override;
    std::shared_ptr<Tensor> loss(
        const std::shared_ptr<Tensor>& input,
        const std::shared_ptr<Tensor>& target
    );
    std::vector<Tensor*> getParams() override;
};
 
