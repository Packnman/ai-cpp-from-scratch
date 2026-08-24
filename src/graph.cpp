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

void Model::save(const char* szFName)
{
    FILE* fp    =fopen( szFName,"wb" );
    if( fp==nullptr )
    {
        throw std::runtime_error(
            "Model::save: failed to open file"
        );
    }

    std::vector<Tensor*> lpmParams  =getParams();
    int nParam  =static_cast<int>( lpmParams.size() );

    fwrite( &nParam,sizeof(int),1,fp );

    for( int i=0;i<nParam;i++ )
    {
        Tensor* lpTensor    =lpmParams[i];
        if( lpTensor==nullptr )
        {
            fclose( fp );
            throw std::runtime_error(
                "Model::save: null parameter"
            );
        }

        int nRows   =lpTensor->_mData._nRows;
        int nCols   =lpTensor->_mData._nCols;
        
        Mat mHost( nRows,nCols );
        lpTensor->_mData.upload( mHost );

        fwrite( mHost._lpfHost,sizeof(float),nRows*nCols,fp );
    }
    
    fclose( fp );
}
void Model::load(const char* szFName)
{
    FILE* fp    =fopen( szFName,"rb" );
    if( fp==nullptr )
    {
        throw std::runtime_error(
            "Model::load: failed to open file"
        );
    }
    //
    std::vector<Tensor*> lpmParams  =getParams();

    int nParam  =0; fread( &nParam,sizeof(int),1,fp );

    if( nParam!=static_cast<int>(lpmParams.size()) )
    {
        fclose( fp );
        throw std::runtime_error(
            "Model::load: parameter count mismatch"
        );
    }
    for( int i=0;i<nParam;i++ )
    {
        Tensor* lpTensor    =lpmParams[i];
        if( lpTensor==nullptr )
        {
            fclose( fp );
            throw std::runtime_error(
                "Model::lead: null parameter"
            );
        }

        int nRows   =0; fread( &nRows,sizeof(int),1,fp );
        int nCols   =0; fread( &nCols,sizeof(int),1,fp );
        if( (nRows!=lpTensor->_mData._nRows)||(nCols!=lpTensor->_mData._nCols) )
        {
            fclose( fp );
            throw std::runtime_error(
                "Model::load: parameter size mismatch"
            );
        }

        Mat mHost( nRows,nCols );   fread( mHost._lpfHost,sizeof(float),nRows*nCols,fp );
        lpTensor->_mData.download( mHost );
    }

    fclose( fp );
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
    std::vector<Tensor*> lpmParams    =getParams();

    for( int i=0;i<static_cast<int>(lpmParams.size());i++ )
    {
        if( lpmParams[i]==nullptr )   {continue;}
        //
        cuda_fill( lpmParams[i]->_mGrad,0.0f );
    }
}