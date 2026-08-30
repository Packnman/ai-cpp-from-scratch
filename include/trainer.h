#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "cuda_tensor.h"
#include "matrix.h"
#include "optimizer.h"

struct TrainConfig {
    int             nEpochs;       // 学習するエポック数
    int             nBatchSize;    // 1バッチあたりのサンプル数
    std::uint32_t   nSeed;         // 乱数生成用のシード値
    float           fLearningRate;   // Optimizerの学習率
};

struct SupervisedBatch {
    std::shared_ptr<Tensor> spmInput;      // 入力データtensor
    std::shared_ptr<Tensor> spmTarget;     // 正解データtensor
    std::size_t             nSize;         // このバッチに含まれるサンプル数
};

struct TrainingState
{
    int nEpoch =0;                     // 現在のエポック番号
    std::uint64_t nGlobalStep =0;      // 学習開始からの累計更新回数
};

namespace trainer_detail {

inline std::vector<std::size_t> makeIndices(std::size_t nSize)
{
    std::vector<std::size_t> nIndices( nSize );
    std::iota( nIndices.begin(),nIndices.end(),0 );

    return nIndices;
}

inline float readScalar(const std::shared_ptr<Tensor>& c_spmTensor)
{
    if( (c_spmTensor==nullptr)||
        (c_spmTensor->_mData._nRows!=1)||
        (c_spmTensor->_mData._nCols!=1) )
    {
        throw std::runtime_error(
            "trainer: loss tensor must have shape 1 x 1"
        );
    }

    Mat mHost( 1,1 );
    c_spmTensor->_mData.upload( mHost );

    return mHost( 0,0 );
}

inline void validateConfig(const TrainConfig& c_cfgConfig)
{
    if( c_cfgConfig.nEpochs<=0 )
    {
        throw std::runtime_error(
            "trainer: epochs must be positive"
        );
    }
    if( c_cfgConfig.nBatchSize<=0 )
    {
        throw std::runtime_error(
            "trainer: batch size must be positive"
        );
    }
}

}  // namespace trainer_detail

template<class ModelType,class DatasetType>
void train(
    ModelType& mdlModel,
    Optimizer& optOptimizer,
    const DatasetType& c_dtsDataset,
    const TrainConfig& c_cfgConfig
)
{
    trainer_detail::validateConfig( c_cfgConfig );
    if( c_dtsDataset.size()==0 )
    {
        throw std::runtime_error(
            "train: dataset is empty"
        );
    }

    std::vector<std::size_t> nIndices =trainer_detail::makeIndices(
        c_dtsDataset.size()
    );
    std::mt19937 rngRandom( c_cfgConfig.nSeed );

    optOptimizer.init();

    for( int nEpoch=0;nEpoch<c_cfgConfig.nEpochs;++nEpoch )
    {
        std::shuffle( nIndices.begin(),nIndices.end(),rngRandom );

        double dblLossSum =0.0;
        std::size_t nSampleCount =0;

        for( std::size_t nBegin=0;
             nBegin<nIndices.size();
             nBegin+=static_cast<std::size_t>(c_cfgConfig.nBatchSize) )
        {
            std::size_t nCount =std::min<std::size_t>(
                static_cast<std::size_t>(c_cfgConfig.nBatchSize),
                nIndices.size()-nBegin
            );
            SupervisedBatch batBatch =c_dtsDataset.makeBatch(
                nIndices,
                nBegin,
                nCount
            );
            if( (batBatch.spmInput==nullptr)||
                (batBatch.spmTarget==nullptr)||
                (batBatch.nSize!=nCount) )
            {
                throw std::runtime_error(
                    "train: dataset returned an invalid batch"
                );
            }

            optOptimizer.zero_grads();

            std::shared_ptr<Tensor> spmLoss =mdlModel.loss(
                batBatch.spmInput,
                batBatch.spmTarget
            );
            spmLoss->backward();

            float fBatchLoss =trainer_detail::readScalar( spmLoss );
            optOptimizer.update();

            dblLossSum +=static_cast<double>(fBatchLoss)*batBatch.nSize;
            nSampleCount +=batBatch.nSize;
        }

        printf(
            "epoch %d/%d  loss=%.6f\n",
            nEpoch + 1,
            c_cfgConfig.nEpochs,
            dblLossSum /static_cast<double>(nSampleCount)
        );
    }
}

template<class ModelType,class DatasetType,class MetricType>
float evaluate(
    ModelType& mdlModel,
    const DatasetType& c_dtsDataset,
    int nBatchSize,
    MetricType metMetric
)
{
    if( nBatchSize<=0 )
    {
        throw std::runtime_error(
            "evaluate: batch size must be positive"
        );
    }
    if( c_dtsDataset.size()==0 )
    {
        throw std::runtime_error(
            "evaluate: dataset is empty"
        );
    }

    std::vector<std::size_t> nIndices =
        trainer_detail::makeIndices( c_dtsDataset.size() );
    std::size_t nCorrect =0;

    for( std::size_t nBegin=0;
         nBegin<nIndices.size();
         nBegin+=static_cast<std::size_t>(nBatchSize) )
    {
        std::size_t nCount =std::min<std::size_t>(
            static_cast<std::size_t>(nBatchSize),
            nIndices.size()-nBegin
        );
        SupervisedBatch batBatch =c_dtsDataset.makeBatch(
            nIndices,
            nBegin,
            nCount
        );
        if( (batBatch.spmInput==nullptr)||
            (batBatch.spmTarget==nullptr)||
            (batBatch.nSize!=nCount) )
        {
            throw std::runtime_error(
                "evaluate: dataset returned an invalid batch"
            );
        }

        std::vector<std::shared_ptr<Tensor>> spmInputs{
            batBatch.spmInput
        };
        std::shared_ptr<Tensor> spmOutput =mdlModel.forward( spmInputs );

        nCorrect +=metMetric(
            spmOutput,
            batBatch.spmTarget
        );
    }

    return static_cast<float>(nCorrect)/
           static_cast<float>(c_dtsDataset.size());
}
