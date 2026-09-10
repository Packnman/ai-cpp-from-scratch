#pragma once

#include "cuda_function.h"


// --------------------------
// BatchMatMul
// --------------------------
class BatchMatMul: public Function
{
public:
    BatchMatMul();
    ~BatchMatMul();

private:
    struct ShapeInfo
    {
        // d_head =d_model / head
        // A [token, d_head, head, batch]
        // B [d_head, token, head, batch]
        int nM;     // A tokens
        int nK;     // d_head
        int nN;     // B tokens
        int nBatch; // head * batch
        std::vector<std::int64_t> shapeOutput;  // output shape
    };

    static ShapeInfo _validateInputs(
        const std::vector<std::shared_ptr<Tensor>>& c_spmInputs,
        const char* c_lpszOperation
    );

public:
    void backward(
        const std::vector<const cufMat*>& c_lpmOutputGrads,
        const std::vector<std::shared_ptr<Tensor>>& c_spmInputs,
        const std::vector<std::shared_ptr<Tensor>>& c_spmOutputs
    ) override;
    std::vector<std::shared_ptr<Tensor>> forward(
        const std::vector<std::shared_ptr<Tensor>>& c_spmInputs
    ) override;
};
