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
    int             epochs;
    int             batchSize;
    std::uint32_t   seed;
};

struct SupervisedBatch {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> target;
    std::size_t             size;
};

namespace trainer_detail {

inline std::vector<std::size_t> makeIndices(std::size_t size)
{
    std::vector<std::size_t> indices( size );
    std::iota( indices.begin(),indices.end(),0 );

    return indices;
}

inline float readScalar(const std::shared_ptr<Tensor>& tensor)
{
    if( (tensor==nullptr)||
        (tensor->_mData._nRows!=1)||
        (tensor->_mData._nCols!=1) )
    {
        throw std::runtime_error(
            "trainer: loss tensor must have shape 1 x 1"
        );
    }

    Mat host( 1,1 );
    tensor->_mData.upload( host );

    return host( 0,0 );
}

inline void validateConfig(const TrainConfig& config)
{
    if( config.epochs<=0 )
    {
        throw std::runtime_error(
            "trainer: epochs must be positive"
        );
    }
    if( config.batchSize<=0 )
    {
        throw std::runtime_error(
            "trainer: batch size must be positive"
        );
    }
}

}  // namespace trainer_detail

template<class ModelType,class DatasetType>
void train(
    ModelType& model,
    Optimizer& optimizer,
    const DatasetType& dataset,
    const TrainConfig& config
)
{
    trainer_detail::validateConfig( config );
    if( dataset.size()==0 )
    {
        throw std::runtime_error(
            "train: dataset is empty"
        );
    }

    std::vector<std::size_t> indices    =trainer_detail::makeIndices( dataset.size() );
    std::mt19937 random( config.seed );

    optimizer.init();

    for( int epoch=0;epoch<config.epochs;epoch++ )
    {
        std::shuffle( indices.begin(),indices.end(),random );

        double lossSum =0.0;
        std::size_t sampleCount =0;

        for( std::size_t begin=0;
             begin<indices.size();
             begin+=static_cast<std::size_t>(config.batchSize) )
        {
            std::size_t count   =std::min<std::size_t>(
                static_cast<std::size_t>(config.batchSize),
                indices.size()-begin
            );
            SupervisedBatch batch   =dataset.makeBatch(
                indices,    // inputs
                begin,      // targets
                count       // size
            );
            if( (batch.input==nullptr)||
                (batch.target==nullptr)||
                (batch.size!=count) )
            {
                throw std::runtime_error(
                    "train: dataset returned an invalid batch"
                );
            }

            optimizer.zero_grads();

            std::shared_ptr<Tensor> loss =model.loss(
                batch.input,
                batch.target
            );
            loss->backward();

            float batchLoss =trainer_detail::readScalar( loss );
            optimizer.update();

            lossSum +=static_cast<double>(batchLoss)*batch.size;
            sampleCount +=batch.size;
        }

        printf(
            "epoch %d/%d  loss=%.6f\n",
            epoch + 1,
            config.epochs,
            lossSum / (double)sampleCount
        );
    }
}

template<class ModelType,class DatasetType,class MetricType>
float evaluate(
    ModelType& model,
    const DatasetType& dataset,
    int batchSize,
    MetricType metric
)
{
    if( batchSize<=0 )
    {
        throw std::runtime_error(
            "evaluate: batch size must be positive"
        );
    }
    if( dataset.size()==0 )
    {
        throw std::runtime_error(
            "evaluate: dataset is empty"
        );
    }

    std::vector<std::size_t> indices =
        trainer_detail::makeIndices( dataset.size() );
    std::size_t nCorrect =0;

    for( std::size_t begin=0;
         begin<indices.size();
         begin+=static_cast<std::size_t>(batchSize) )
    {
        std::size_t count =std::min<std::size_t>(
            static_cast<std::size_t>(batchSize),
            indices.size()-begin
        );
        SupervisedBatch batch =dataset.makeBatch(
            indices,
            begin,
            count
        );
        if( (batch.input==nullptr)||
            (batch.target==nullptr)||
            (batch.size!=count) )
        {
            throw std::runtime_error(
                "evaluate: dataset returned an invalid batch"
            );
        }

        std::vector<std::shared_ptr<Tensor>> inputs{
            batch.input
        };
        std::shared_ptr<Tensor> output =model.forward( inputs );

        nCorrect +=metric(
            output,
            batch.target
        );
    }

    return static_cast<float>(nCorrect)/
           static_cast<float>(dataset.size());
}
