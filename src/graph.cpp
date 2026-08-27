#include <iostream>
#include <stdexcept>
#include "matrix.h"
#include "tensor.h"
#include "graph.h"


Graph::Graph()
{

}
Graph::~Graph()
{

}

void Graph::zero_grads()
{
    std::vector<Tensor*> lpmParams =getParams();

    for( int i=0;i<static_cast<int>(lpmParams.size());i++ )
    {
        if( lpmParams[i]==nullptr )    {continue;}
        //
        cuda_fill( lpmParams[i]->_mGrad,0.0f );
    }
}
void Graph::reset_state()
{
    // デフォルトは何もしない
}

std::shared_ptr<Tensor> Graph::forward(
    std::vector<std::shared_ptr<Tensor>>& inputs
)
{
    throw std::runtime_error(
        "Graph::forward is not implumented"
    );
}
// 学習パラメータを返す
std::vector<Tensor*> Graph::getParams()
{
    return std::vector<Tensor*>();
}




// --------------------------
// Model
// --------------------------
Model::Model()
{

}
Model::~Model()
{

}

void Model::zero_grads()
{
    std::vector<Tensor*> params =getParams();

    for( int i=0;i<static_cast<int>(params.size());i++ )
    {
        if( params[i]==nullptr )    {continue;}

        cuda_fill(
            params[i]->_mGrad,
            0.0f
        );
    }
}
void Model::reset_state()
{

}
// 順伝播
std::shared_ptr<Tensor> Model::forward(
    std::vector<std::shared_ptr<Tensor>>& inputs
)
{
    throw std::runtime_error(
        "Model::forward is not implemented"
    );
}
// モデル全体の学習パラメータを返す
std::vector<Tensor*> Model::getParams()
{
    return {};
}