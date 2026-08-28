#include <iostream>
#include <stdio.h>
#include "cuda_function.h"
#include "cuda_tensor.h"


Tensor::Tensor(int rows,int cols)
    :_mData( rows,cols ),
     _mGrad( rows,cols ),
     _spContexts( nullptr )
{
    cuda_fill( _mGrad,0.0f );
}
Tensor::~Tensor()
{
    
}
void Tensor::backward()
{
    // 最終Tensorの勾配 L に対する L 自身の微分なので
    // dL/dL = 1
    //
    cuda_fill( _mGrad,1.0f );   // seed

    std::vector<Context*> contexts;
    std::unordered_set<Context*> visited;

    buildBackwardGraph( this,contexts,visited );

    // 出力側 -> 入力側
    for( int i=static_cast<int>(contexts.size())-1;i>=0;i-- )
    {
        Context* lpContext  =contexts[i];

        // Functionの出力Tensorのgradを取得する
        // 1出力FUnctionを前提とするなら_wpOutputs[0]でよい
        //
        if( lpContext->_lpFunc==nullptr )   {continue;}
        // weak_ptr → shared_ptr
        std::vector<std::shared_ptr<Tensor>> spOutputs;
        for( int j=0;j<static_cast<int>(lpContext->_wpOutputs.size());j++ )
        {
            std::shared_ptr<Tensor> spOutput  =lpContext->_wpOutputs[j].lock();
            if( spOutput!=nullptr )
            {
                spOutputs.push_back( spOutput );
            }
        }
        //
        if( !spOutputs.empty() )
        {
            lpContext->_lpFunc->backward(
                spOutputs[0]->_mGrad,
                lpContext->_spInputs,
                spOutputs
            );
        }
    }
}
void Tensor::buildBackwardGraph(
    Tensor* value,
    std::vector<Context*>& contexts,
    std::unordered_set<Context*>& visited
)
{
    if( value==nullptr )                        {return;}

    Context* lpContext  =value->_spContexts.get();
    if( lpContext==nullptr )                    {return;}
    if( visited.find(lpContext)!=visited.end() ){return;}   // 同じFunitonを二重登録しない
    //
    visited.insert( lpContext );
    // 入力側をたどる
    for( int i=0;i<lpContext->_spInputs.size();i++ )
    {
        buildBackwardGraph(
            lpContext->_spInputs[i].get(),
            contexts,
            visited
        );
    }
    // 入力側を登録した後で自分を登録
    contexts.push_back( lpContext );
}