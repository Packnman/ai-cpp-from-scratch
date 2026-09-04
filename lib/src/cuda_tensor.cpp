#include <iostream>
#include <stdio.h>
#include "cuda_function.h"
#include "cuda_tensor.h"


Tensor::Tensor(int nRows,int nCols)
    :_mData( nRows,nCols ),
     _mGrad( nRows,nCols ),
     _spContext( nullptr )
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

    std::vector<Context*> lpContexts;
    std::unordered_set<Context*> lpVisited;

    buildBackwardGraph( this,lpContexts,lpVisited );

    // 出力側 -> 入力側
    for( int nContext=static_cast<int>(lpContexts.size())-1;
         nContext>=0;
         --nContext )
    {
        Context* lpContext  =lpContexts[nContext];

        if( lpContext->_lpFunc==nullptr )   {continue;}

        // 出力番号を保ったまま、それぞれの出力勾配をFunctionへ渡す。
        // 破棄済みの出力はnullptrとなり、勾配なしとして扱う。
        std::vector<std::shared_ptr<Tensor>> spmOutputs( lpContext->_wpmOutputs.size() );
        std::vector<const cuMat*> lpmOutputGrads(
            lpContext->_wpmOutputs.size(),
            nullptr
        );
        bool isOutputAvailable =false;
        for( std::size_t nOutput=0;
             nOutput<lpContext->_wpmOutputs.size();
             ++nOutput )
        {
            spmOutputs[nOutput] =lpContext->_wpmOutputs[nOutput].lock();
            if( spmOutputs[nOutput]!=nullptr )
            {
                lpmOutputGrads[nOutput] =&spmOutputs[nOutput]->_mGrad;
                isOutputAvailable =true;
            }
        }
        if( isOutputAvailable )
        {
            lpContext->_lpFunc->backward(
                lpmOutputGrads,
                lpContext->_spmInputs,
                spmOutputs
            );
        }
    }
}
void Tensor::buildBackwardGraph(
    Tensor* lpValue,
    std::vector<Context*>& lpContexts,
    std::unordered_set<Context*>& lpVisited
)
{
    if( lpValue==nullptr )                        {return;}

    Context* lpContext  =lpValue->_spContext.get();
    if( lpContext==nullptr )                    {return;}
    if( lpVisited.find(lpContext)!=lpVisited.end() ){return;} // 同じFunctionを二重登録しない
    //
    lpVisited.insert( lpContext );
    // 入力側をたどる
    for( std::size_t nInput=0;
         nInput<lpContext->_spmInputs.size();
         ++nInput )
    {
        buildBackwardGraph(
            lpContext->_spmInputs[nInput].get(),
            lpContexts,
            lpVisited
        );
    }
    // 入力側を登録した後で自分を登録
    lpContexts.push_back( lpContext );
}
