#include <iostream>
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include "matrix.h"



Mat::Mat(int nRows,int nCols)
    :_nRows( nRows ),
     _nCols( nCols ),
     _lpfHost( (float*)malloc(nRows*nCols*sizeof(float)) )
{
    
}
Mat::Mat(const Mat& val)
    :_nRows( val._nRows ),
     _nCols( val._nCols ),
     _lpfHost( (float*)malloc(val._nRows*val._nCols*sizeof(float)) )
{
    memcpy( _lpfHost,val._lpfHost,_nRows*_nCols*sizeof(float) );
}
Mat::Mat(Mat&& val) noexcept
    :_nRows( val._nRows ),
     _nCols( val._nCols ),
     _lpfHost( val._lpfHost )
{
    val._nRows      =0;
    val._nCols      =0;
    val._lpfHost    =nullptr;
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
float Mat::tri() const
{
    if( (_nRows!=_nCols) )
    {
        throw std::runtime_error(
            "Mat::tri: matrix size missmatch"
        );
    }
    //
    float rst   =0.0f;

    for( int i=0;i<_nRows;i++ )
    {
        rst +=(*this)(i,i);
    }

    return rst;
}
Mat Mat::trp() const
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
Euler Mat::toEul() const
{
    if( (_nRows!=3)||(_nCols!=3) )
    {
        throw std::runtime_error(
            "Mat::toEul: matrix size missmatch"
        );
    }
    //
    Euler rst;
    float sp    =(*this)(0,2);

    if( sp>1.0f )   {sp =1.0f;}
    if( sp<-1.0f )  {sp =-1.0f;}
    rst(1)  =asinf( -sp );
    rst(2)  =atan2f( (*this)(0,1),(*this)(0,0) );
    rst(0)  =atan2f( (*this)(1,2),(*this)(2,2) );

    return rst;
}
Quaternion Mat::toQtn() const
{
    if( (_nRows!=3)||(_nCols!=3) )
    {
        throw std::runtime_error(
            "Mat::toQtn: matrix size missmatch"
        );
    }
    //
    Quaternion rst;

    float tr    =(*this).tri();
    float rad   =acosf( (tr-1.0)/2.0 );
    if( fabsf(sinf(rad))<1.0e-6f )
    {
        // 特異点処理
        rst(0)  =0.0;
        rst(1)  =0.0;
        rst(2)  =0.0;
        rst(3)  =1.0;
    }
    else
    {
        rst(0)  =( (*this)(1,2)-(*this)(2,1) )/(2.0*sinf(rad))*sinf(rad/2.0);
        rst(1)  =( (*this)(2,0)-(*this)(0,2) )/(2.0*sinf(rad))*sinf(rad/2.0);
        rst(2)  =( (*this)(0,1)-(*this)(1,0) )/(2.0*sinf(rad))*sinf(rad/2.0);
        rst(3)  =cosf( rad/2.0 );
    }
    rst.normalized();
    
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
        for( int j=0;j<val._nCols;j++ )
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


// --------------------------
// Vector
// --------------------------
Vec::Vec(int nRows)
    :Mat(nRows,1)
{

}
Vec::Vec(const Mat& val)
    :Mat(val)
{
    if( val._nCols!=1 )
    {
        throw std::runtime_error(
            "Vec::Vec: matrix must have exactly one column"
        );
    }
}
Vec::~Vec()
{

}
Mat Vec::skew() const
{
    if( _nRows!=3 )
    {
        throw std::runtime_error(
            "Vec::skew: vector size missmatch"
        );
    }
    //
    Mat rst(3,3);

    rst(0,0) = 0.0f;
    rst(0,1) =-(*this)(2);
    rst(0,2) = (*this)(1);
    rst(1,0) = (*this)(2);
    rst(1,1) = 0.0f;
    rst(1,2) =-(*this)(0);
    rst(2,0) =-(*this)(1);
    rst(2,1) = (*this)(0);
    rst(2,2) = 0.0f;

    return rst;
}
Vec Vec::cpx(const Vec& val) const
{
    Vec rst(3);

    if( (_nRows!=3)||(val._nRows!=3) )
    {
        throw std::runtime_error(
            "Vec::cpx: vector size missmatch"
        );
    }
    //
    rst(0)  =(*this)(1)*val(2) - (*this)(2)*val(1);
    rst(1)  =(*this)(2)*val(0) - (*this)(0)*val(2);
    rst(2)  =(*this)(0)*val(1) - (*this)(1)*val(0);

    return rst;
}
float Vec::dot(const Vec& val) const
{
    float rst   =0.0f;

    if( (_nRows!=val._nRows) )
    {
        throw std::runtime_error(
            "Vec::dot: vector size missmatch"
        );
    }
    //
    for( int i=0;i<_nRows;i++ )
    {
        rst +=(*this)(i)*val(i);
    }

    return rst;
}
float Vec::norm() const
{
    return sqrtf( (*this).dot((*this)) );
}
Quaternion Vec::exp(float dt) const
{
    if( _nRows!=3 )
    {
        throw std::runtime_error(
            "Vec::exp: vector size missmatch"
        );
    }
    //
    Quaternion rst;
    //
    float omg   =(*this).norm();
    Vec rps( *this );
    rps(0)  /=omg;
    rps(1)  /=omg;
    rps(2)  /=omg;
    //
    rst(0)   =rps(0)*sinf( 0.5*omg*dt );
    rst(1)   =rps(1)*sinf( 0.5*omg*dt );
    rst(2)   =rps(2)*sinf( 0.5*omg*dt );
    rst(3)   =cosf( 0.5*omg*dt );
    rst.normalized();

    return rst;
}
float& Vec::operator()(int row)
{
    return Mat::operator()(row,0);
}
const float& Vec::operator()(int row) const
{
    return Mat::operator()(row,0);
}
Vec operator*(const Mat& L,const Vec& R)
{
    const Mat& mR   =static_cast<const Mat&>(R);

    return L*mR;
}

// --------------------------
// Euler
// --------------------------
Euler::Euler()
    :Vec(3)
{

}
Euler::Euler(const Mat& val)
    :Vec(val)
{
    if( val._nRows!=3 )
    {
        throw std::runtime_error(
            "Euler::Euler: vector size must be 3"
        );
    }
}
Euler::~Euler()
{

}
Mat Euler::toDCM() const
{
    Mat rst(3,3);

    rst(0,0)    = cosf((*this)(2))*cosf((*this)(1));
    rst(0,1)    = sinf((*this)(2))*cosf((*this)(1));
    rst(0,2)    =-sinf((*this)(1));
    rst(1,0)    = cosf((*this)(2))*sinf((*this)(1))*sinf((*this)(0))-sinf((*this)(2))*cosf((*this)(0));
    rst(1,1)    = sinf((*this)(2))*sinf((*this)(1))*sinf((*this)(0))+cosf((*this)(2))*cosf((*this)(0));
    rst(1,2)    = cosf((*this)(1))*sinf((*this)(0));
    rst(2,0)    = cosf((*this)(2))*sinf((*this)(1))*cosf((*this)(0))+sinf((*this)(2))*sinf((*this)(0));
    rst(2,1)    = sinf((*this)(2))*sinf((*this)(1))*cosf((*this)(0))-cosf((*this)(2))*sinf((*this)(0));
    rst(2,2)    = cosf((*this)(1))*cosf((*this)(0));

    return rst;
}
Quaternion Euler::toQtn() const
{
    return (*this).toDCM().toQtn();
}
Euler operator*(const Euler& e1,const Euler& e2)
{
    return (e1.toDCM()*e2.toDCM()).toEul();
}
Vec operator*(const Euler& e,const Vec& v)
{
    return e.toDCM()*v;
}

// --------------------------
// Quaternion
// --------------------------
Quaternion::Quaternion()
    :Vec(4)
{

}
Quaternion::Quaternion(const Mat& val)
    :Vec(val)
{
    if( val._nRows!=4 )
    {
        throw std::runtime_error(
            "Quaternion::Quaternion: vector size must be 4"
        );
    }
}
Quaternion::~Quaternion()
{

}
Quaternion Quaternion::cnj() const
{
    Quaternion rst;

    rst(0)  =-(*this)(0);
    rst(1)  =-(*this)(1);
    rst(2)  =-(*this)(2);
    rst(3)  = (*this)(3);

    return rst;
}
Vec Quaternion::axis() const
{
    Vec rst(3);

    rst(0)  =(*this)(0);
    rst(1)  =(*this)(1);
    rst(2)  =(*this)(2);

    return rst;
}
Mat Quaternion::dot_E() const
{
    Mat rst(4,3);

    rst(0,0)    = (*this)(3);
    rst(0,1)    =-(*this)(2);
    rst(0,2)    = (*this)(1);
    rst(1,0)    = (*this)(2);
    rst(1,1)    = (*this)(3);
    rst(1,2)    =-(*this)(0);
    rst(2,0)    =-(*this)(1);
    rst(2,1)    = (*this)(0);
    rst(2,2)    = (*this)(3);
    rst(3,0)    = (*this)(0);
    rst(3,1)    =-(*this)(1);
    rst(3,2)    =-(*this)(2);

    return rst;
}
Quaternion Quaternion::dot(const Vec& rps) const
{
    Quaternion rst;

    Mat Eq =(*this).dot_E();

    rst(0)  =Eq(0,0)*rps(0) + Eq(0,1)*rps(1) + Eq(0,2)*rps(2);
    rst(1)  =Eq(1,0)*rps(0) + Eq(1,1)*rps(1) + Eq(1,2)*rps(2);
    rst(2)  =Eq(2,0)*rps(0) + Eq(2,1)*rps(1) + Eq(2,2)*rps(2);
    rst(3)  =Eq(3,0)*rps(0) + Eq(3,1)*rps(1) + Eq(3,2)*rps(2);

    return rst;
}
void Quaternion::normalized()
{
    float norm  =(*this).norm();

    if( norm<=0.0f )
    {
        throw std::runtime_error(
            "Quaternion::normalized: zero norm"
        );
    }
    //
    (*this)(0)  /=norm;
    (*this)(1)  /=norm;
    (*this)(2)  /=norm;
    (*this)(3)  /=norm;
}
Mat Quaternion::toDCM() const
{
    Mat rst(3,3);

    rst(0,0)    =1.0-2.0*((*this)(1)*(*this)(1)+(*this)(2)*(*this)(2));
    rst(0,1)    =2.0*((*this)(0)*(*this)(1) + (*this)(2)*(*this)(3));
    rst(0,2)    =2.0*((*this)(2)*(*this)(0) - (*this)(1)*(*this)(3));
    rst(1,0)    =2.0*((*this)(0)*(*this)(1) - (*this)(2)*(*this)(3));
    rst(1,1)    =1.0-2.0*((*this)(0)*(*this)(0)+(*this)(2)*(*this)(2));
    rst(1,2)    =2.0*((*this)(1)*(*this)(2) + (*this)(0)*(*this)(3));
    rst(2,0)    =2.0*((*this)(2)*(*this)(0) + (*this)(1)*(*this)(3));
    rst(2,1)    =2.0*((*this)(1)*(*this)(2) - (*this)(0)*(*this)(3));
    rst(2,2)    =1.0-2.0*((*this)(0)*(*this)(0)+(*this)(1)*(*this)(1));

    return rst;
}
Euler Quaternion::toEul() const
{
    return (*this).toDCM().toEul();
}
Quaternion operator*(const Quaternion& q1,const Quaternion& q2)
{
    Quaternion rst;

    Vec qv1         =q1.axis();
    Vec qv2         =q2.axis();
    Vec q2qv1       =qv1*q2(3);
    Vec q1qv2       =qv2*q1(3);
    Vec qv2xqv1     =qv2.cpx( qv1 );
    float qv2qv1    =qv2.dot( qv1 );

    rst(0)  =q2qv1(0)+q1qv2(0)+qv2xqv1(0);
    rst(1)  =q2qv1(1)+q1qv2(1)+qv2xqv1(1);
    rst(2)  =q2qv1(2)+q1qv2(2)+qv2xqv1(2);
    rst(3)  =q2(3)*q1(3)-qv2qv1;
    rst.normalized();

    return rst;
}
Vec operator*(const Quaternion& q,const Vec& v)
{
    return q.toDCM()*v;
}
