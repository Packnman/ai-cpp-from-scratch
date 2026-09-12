#pragma once

#include "cuda_function.h"

// A graph-local instance owns an immutable target snapshot and probabilities.
class IndexCrossEntropy final : public Function
{
public:
    IndexCrossEntropy( const cunMat& c_mTargets, int nPadId = 0 );
    TensorList forward( const TensorList& c_spmInputs ) override;
    void backward( const TensorGradList& c_lpmOutputGrads, const TensorList& c_spmInputs,
                   const TensorList& c_spmOutputs ) override;

private:
    cunMat _mTargets;
    cufMat _mProbabilities;
    int _nPadId;
    int _nValid;
    bool _isUsed = false;
};
