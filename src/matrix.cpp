#include <iostream>
#include <stdio.h>
#include <memory.h>
#include "matrix.h"



Mat::Mat(int nRows,int nCols)
{
    _nRows      =nRows;
    _nCols      =nCols;
    _lpfHost    =(float*)malloc( _nRows*_nCols*sizeof(float) );
}
Mat::Mat(const Mat& val)
{
    _nRows      =val._nRows;
    _nCols      =val._nCols;
    _lpfHost    =(float*)malloc( _nRows*_nCols*sizeof(float) );

    memcpy( _lpfHost,val._lpfHost,_nRows*_nCols*sizeof(float) );
}
Mat::~Mat()
{
    free( _lpfHost );
    _lpfHost    =nullptr;
}
void Mat::ones()
{
    for( int i=0;i<_nRows;i++ )
    {
        for( int j=0;j<_nCols;j++ )
        {
            (*this)(i,j)    =1.0f;
        }
    }
}
Mat Mat::transpose()
{
    Mat rst(_nCols,_nRows);
    
    for( int i=0;i<rst._nRows;i++ )
    {
        for( int j=0;j<rst._nCols;j++ )
        {
            rst(i,j)    =(*this)(j,i);
        }
    }

    return rst;
}
Mat& Mat::operator=(const Mat& val)
{
    if( this==&val )    {return *this;}
    if( (_nRows!=val._nRows)||(_nCols!=val._nCols) )
    {
        free( _lpfHost );
        //
        _nRows      =val._nRows;
        _nCols      =val._nCols;
        _lpfHost    =(float*)malloc( _nRows*_nCols*sizeof(float) );
    }

    memcpy( _lpfHost,val._lpfHost,_nRows*_nCols*sizeof(float) );

    return *this;
}
Mat& Mat::operator=(Mat&& val) noexcept
{
    if( this==&val )    {return *this;}

    free( _lpfHost );
    //
    _nRows      =val._nRows;
    _nCols      =val._nCols;
    _lpfHost    =val._lpfHost;

    val._nRows      =0;
    val._nCols      =0;
    val._lpfHost    =nullptr;

    memcpy( _lpfHost,val._lpfHost,_nRows*_nCols*sizeof(float) );

    return *this;
}
Mat operator+(const Mat& L,const Mat& R)
{
    // 行列数の確認
    if( (L._nRows!=R._nRows)||(L._nCols!=R._nCols) )
    {
        throw std::runtime_error(
            std::string( "Mat operator+: matrix size missmatch\n" )
        );
    }
    //
    Mat rst(L);
    rst +=R;

    return rst;
}
Mat operator-(const Mat& L,const Mat& R)
{
    // 行列数の確認
    if( (L._nRows!=R._nRows)||(L._nCols!=R._nCols) )
    {
        throw std::runtime_error(
            std::string( "Mat operator-: matrix size missmatch\n" )
        );
    }
    //
    Mat rst(L);
    rst -=R;

    return rst;
}
Mat operator*(const Mat& L,const Mat& R)
{
    // 行列数の確認
    if( (L._nCols!=R._nRows) )
    {
        throw std::runtime_error(
            std::string( "Mat operator*: matrix size missmatch\n" )
        );
    }
    //
    Mat rst(L._nRows, R._nCols);
    for (int i = 0; i < L._nRows; i++)
    {
        for (int j = 0; j < R._nCols; j++)
        {
            float sum   =0.0f;
            for (int k = 0; k < L._nCols; k++)
            {
                sum     +=L(i,k) * R(k,j);
            }
            rst(i,j) = sum;
        }
    }

    return rst;
}
Mat& Mat::operator+=(const Mat& val)
{
    // 行列数の確認
    if( (_nRows!=val._nRows)||(_nCols!=val._nCols) )
    {
        throw std::runtime_error(
            std::string( "Mat operator+=: matrix size missmatch\n" )
        );
    }
    //
    for( int i=0;i<_nRows;i++ )
    {
        for( int j=0;j<_nCols;j++ )
        {
            (*this)(i,j) +=val(i,j);
        }
    }

    return *this;
}
Mat& Mat::operator-=(const Mat& val)
{
    // 行列数の確認
    if( (_nRows!=val._nRows)||(_nCols!=val._nCols) )
    {
        throw std::runtime_error(
            std::string( "Mat operator-=: matrix size missmatch\n" )
        );
    }
    //
    for( int i=0;i<_nRows;i++ )
    {
        for( int j=0;j<_nCols;j++ )
        {
            (*this)(i,j) -=val(i,j);
        }
    }

    return *this;
}
Mat& Mat::operator*=(const Mat& val)
{
    // 行列数の確認
    if( (_nCols!=val._nRows) )
    {
        throw std::runtime_error(
            std::string( "Mat operator*: matrix size missmatch\n" )
        );
    }
    //
    Mat tmp(_nRows,val._nCols);
    for( int i=0;i<_nRows;i++ )
    {
        for( int j=0;j<_nCols;j++ )
        {
            float sum   =0.0f;
            for( int k=0;k<_nCols;k++ )
            {
                sum +=(*this)(i,k) * val(k,j);
            }
            tmp(i,j)    =sum;
        }
    }

    *this = std::move(tmp);

    return *this;
}
Mat& Mat::operator*=(float val)
{
    for( int i=0;i<_nRows;i++ )
    {
        for( int j=0;j<_nCols;j++ )
        {
            (*this)(i,j)    *=val;
        }
    }

    return *this;
}
float& Mat::operator()(int row,int col)
{
    return _lpfHost[IDX2F(row,col,_nRows)];
}
const float& Mat::operator()(int row,int col) const
{
    return _lpfHost[IDX2F(row,col,_nRows)];
}