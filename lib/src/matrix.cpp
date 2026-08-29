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
Mat::Mat(const Mat& c_mValue)
    :_nRows( c_mValue._nRows ),
     _nCols( c_mValue._nCols ),
     _lpfHost( (float*)malloc(c_mValue._nRows*c_mValue._nCols*sizeof(float)) )
{
    memcpy( _lpfHost,c_mValue._lpfHost,_nRows*_nCols*sizeof(float) );
}
Mat::Mat(Mat&& mValue) noexcept
    :_nRows( mValue._nRows ),
     _nCols( mValue._nCols ),
     _lpfHost( mValue._lpfHost )
{
    mValue._nRows      =0;
    mValue._nCols      =0;
    mValue._lpfHost    =nullptr;
}
Mat::~Mat()
{
    free( _lpfHost );
    _lpfHost    =nullptr;
}
void Mat::ones()
{
    for( int nRow=0;nRow<_nRows;nRow++ )
    {
        for( int nCol=0;nCol<_nCols;nCol++ )
        {
            (*this)(nRow,nCol)    =1.0f;
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
    float fResult =0.0f;

    for( int nRow=0;nRow<_nRows;nRow++ )
    {
        fResult +=(*this)(nRow,nRow);
    }

    return fResult;
}
Mat Mat::trp() const
{
    Mat mResult(_nCols,_nRows);
    
    for( int nRow=0;nRow<mResult._nRows;++nRow )
    {
        for( int nCol=0;nCol<mResult._nCols;++nCol )
        {
            mResult(nRow,nCol) =(*this)(nCol,nRow);
        }
    }

    return mResult;
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
    Euler eulResult;
    float fSinPitch =(*this)(0,2);

    if( fSinPitch>1.0f )   {fSinPitch =1.0f;}
    if( fSinPitch<-1.0f )  {fSinPitch =-1.0f;}
    eulResult(1) =asinf( -fSinPitch );
    eulResult(2) =atan2f( (*this)(0,1),(*this)(0,0) );
    eulResult(0) =atan2f( (*this)(1,2),(*this)(2,2) );

    return eulResult;
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
    Quaternion qtnResult;

    float fTrace =(*this).tri();
    float fRadians =acosf( (fTrace-1.0f)/2.0f );
    if( fabsf(sinf(fRadians))<1.0e-6f )
    {
        // 特異点処理
        qtnResult(0) =0.0f;
        qtnResult(1) =0.0f;
        qtnResult(2) =0.0f;
        qtnResult(3) =1.0f;
    }
    else
    {
        qtnResult(0) =( (*this)(1,2)-(*this)(2,1) )/
            (2.0f*sinf(fRadians))*sinf(fRadians/2.0f);
        qtnResult(1) =( (*this)(2,0)-(*this)(0,2) )/
            (2.0f*sinf(fRadians))*sinf(fRadians/2.0f);
        qtnResult(2) =( (*this)(0,1)-(*this)(1,0) )/
            (2.0f*sinf(fRadians))*sinf(fRadians/2.0f);
        qtnResult(3) =cosf( fRadians/2.0f );
    }
    qtnResult.normalized();
    
    return qtnResult;
}
Mat& Mat::operator=(const Mat& c_mValue)
{
    if( this==&c_mValue )    {return *this;}
    if( (_nRows!=c_mValue._nRows)||(_nCols!=c_mValue._nCols) )
    {
        free( _lpfHost );
        //
        _nRows      =c_mValue._nRows;
        _nCols      =c_mValue._nCols;
        _lpfHost    =(float*)malloc( _nRows*_nCols*sizeof(float) );
    }

    memcpy( _lpfHost,c_mValue._lpfHost,_nRows*_nCols*sizeof(float) );

    return *this;
}
Mat& Mat::operator=(Mat&& mValue) noexcept
{
    if( this==&mValue )    {return *this;}

    free( _lpfHost );
    //
    _nRows      =mValue._nRows;
    _nCols      =mValue._nCols;
    _lpfHost    =mValue._lpfHost;

    mValue._nRows      =0;
    mValue._nCols      =0;
    mValue._lpfHost    =nullptr;
    
    return *this;
}
Mat operator+(const Mat& c_mL,const Mat& c_mR)
{
    // 行列数の確認
    if( (c_mL._nRows!=c_mR._nRows)||(c_mL._nCols!=c_mR._nCols) )
    {
        throw std::runtime_error(
            std::string( "Mat operator+: matrix size missmatch\n" )
        );
    }
    //
    Mat mResult(c_mL);
    mResult +=c_mR;

    return mResult;
}
Mat operator-(const Mat& c_mL,const Mat& c_mR)
{
    // 行列数の確認
    if( (c_mL._nRows!=c_mR._nRows)||(c_mL._nCols!=c_mR._nCols) )
    {
        throw std::runtime_error(
            std::string( "Mat operator-: matrix size missmatch\n" )
        );
    }
    //
    Mat mResult(c_mL);
    mResult -=c_mR;

    return mResult;
}
Mat operator*(const Mat& c_mL,const Mat& c_mR)
{
    // 行列数の確認
    if( (c_mL._nCols!=c_mR._nRows) )
    {
        throw std::runtime_error(
            std::string( "Mat operator*: matrix size missmatch\n" )
        );
    }
    //
    Mat mResult(c_mL._nRows,c_mR._nCols);
    for( int nRow=0;nRow<c_mL._nRows;++nRow )
    {
        for( int nCol=0;nCol<c_mR._nCols;++nCol )
        {
            float fSum =0.0f;
            for( int nInner=0;nInner<c_mL._nCols;++nInner )
            {
                fSum +=c_mL(nRow,nInner)*c_mR(nInner,nCol);
            }
            mResult(nRow,nCol) =fSum;
        }
    }

    return mResult;
}
Mat& Mat::operator+=(const Mat& c_mValue)
{
    // 行列数の確認
    if( (_nRows!=c_mValue._nRows)||(_nCols!=c_mValue._nCols) )
    {
        throw std::runtime_error(
            std::string( "Mat operator+=: matrix size missmatch\n" )
        );
    }
    //
    for( int nRow=0;nRow<_nRows;nRow++ )
    {
        for( int nCol=0;nCol<_nCols;nCol++ )
        {
            (*this)(nRow,nCol) +=c_mValue(nRow,nCol);
        }
    }

    return *this;
}
Mat& Mat::operator-=(const Mat& c_mValue)
{
    // 行列数の確認
    if( (_nRows!=c_mValue._nRows)||(_nCols!=c_mValue._nCols) )
    {
        throw std::runtime_error(
            std::string( "Mat operator-=: matrix size missmatch\n" )
        );
    }
    //
    for( int nRow=0;nRow<_nRows;nRow++ )
    {
        for( int nCol=0;nCol<_nCols;nCol++ )
        {
            (*this)(nRow,nCol) -=c_mValue(nRow,nCol);
        }
    }

    return *this;
}
Mat& Mat::operator*=(const Mat& c_mValue)
{
    // 行列数の確認
    if( (_nCols!=c_mValue._nRows) )
    {
        throw std::runtime_error(
            std::string( "Mat operator*: matrix size missmatch\n" )
        );
    }
    //
    Mat mTemporary(_nRows,c_mValue._nCols);
    for( int nRow=0;nRow<_nRows;nRow++ )
    {
        for( int nCol=0;nCol<c_mValue._nCols;++nCol )
        {
            float fSum =0.0f;
            for( int nInner=0;nInner<_nCols;++nInner )
            {
                fSum +=(*this)(nRow,nInner)*c_mValue(nInner,nCol);
            }
            mTemporary(nRow,nCol) =fSum;
        }
    }

    *this =std::move(mTemporary);

    return *this;
}
Mat& Mat::operator*=(float fValue)
{
    for( int nRow=0;nRow<_nRows;nRow++ )
    {
        for( int nCol=0;nCol<_nCols;nCol++ )
        {
            (*this)(nRow,nCol) *=fValue;
        }
    }

    return *this;
}
float& Mat::operator()(int nRow,int nCol)
{
    return _lpfHost[IDX2F(nRow,nCol,_nRows)];
}
const float& Mat::operator()(int nRow,int nCol) const
{
    return _lpfHost[IDX2F(nRow,nCol,_nRows)];
}


// --------------------------
// Vector
// --------------------------
Vec::Vec(int nRows)
    :Mat(nRows,1)
{

}
Vec::Vec(const Mat& c_mValue)
    :Mat(c_mValue)
{
    if( c_mValue._nCols!=1 )
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
    Mat mResult(3,3);

    mResult(0,0) = 0.0f;
    mResult(0,1) =-(*this)(2);
    mResult(0,2) = (*this)(1);
    mResult(1,0) = (*this)(2);
    mResult(1,1) = 0.0f;
    mResult(1,2) =-(*this)(0);
    mResult(2,0) =-(*this)(1);
    mResult(2,1) = (*this)(0);
    mResult(2,2) = 0.0f;

    return mResult;
}
Vec Vec::cpx(const Vec& c_vValue) const
{
    Vec vResult(3);

    if( (_nRows!=3)||(c_vValue._nRows!=3) )
    {
        throw std::runtime_error(
            "Vec::cpx: vector size missmatch"
        );
    }
    //
    vResult(0) =(*this)(1)*c_vValue(2)-(*this)(2)*c_vValue(1);
    vResult(1) =(*this)(2)*c_vValue(0)-(*this)(0)*c_vValue(2);
    vResult(2) =(*this)(0)*c_vValue(1)-(*this)(1)*c_vValue(0);

    return vResult;
}
float Vec::dot(const Vec& c_vValue) const
{
    float fResult =0.0f;

    if( (_nRows!=c_vValue._nRows) )
    {
        throw std::runtime_error(
            "Vec::dot: vector size missmatch"
        );
    }
    //
    for( int nRow=0;nRow<_nRows;nRow++ )
    {
        fResult +=(*this)(nRow)*c_vValue(nRow);
    }

    return fResult;
}
float Vec::norm() const
{
    return sqrtf( (*this).dot((*this)) );
}
Quaternion Vec::exp(float fDt) const
{
    if( _nRows!=3 )
    {
        throw std::runtime_error(
            "Vec::exp: vector size missmatch"
        );
    }
    //
    Quaternion qtnResult;
    //
    float fOmega =(*this).norm();
    Vec vRps( *this );
    vRps(0) /=fOmega;
    vRps(1) /=fOmega;
    vRps(2) /=fOmega;
    //
    qtnResult(0) =vRps(0)*sinf( 0.5f*fOmega*fDt );
    qtnResult(1) =vRps(1)*sinf( 0.5f*fOmega*fDt );
    qtnResult(2) =vRps(2)*sinf( 0.5f*fOmega*fDt );
    qtnResult(3) =cosf( 0.5f*fOmega*fDt );
    qtnResult.normalized();

    return qtnResult;
}
float& Vec::operator()(int nRow)
{
    return Mat::operator()(nRow,0);
}
const float& Vec::operator()(int nRow) const
{
    return Mat::operator()(nRow,0);
}
Vec operator*(const Mat& c_mL,const Vec& c_vR)
{
    const Mat& c_mR =static_cast<const Mat&>(c_vR);

    return c_mL*c_mR;
}

// --------------------------
// Euler
// --------------------------
Euler::Euler()
    :Vec(3)
{

}
Euler::Euler(const Mat& c_mValue)
    :Vec(c_mValue)
{
    if( c_mValue._nRows!=3 )
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
    Mat mResult(3,3);

    mResult(0,0) = cosf((*this)(2))*cosf((*this)(1));
    mResult(0,1) = sinf((*this)(2))*cosf((*this)(1));
    mResult(0,2) =-sinf((*this)(1));
    mResult(1,0) = cosf((*this)(2))*sinf((*this)(1))*sinf((*this)(0))-sinf((*this)(2))*cosf((*this)(0));
    mResult(1,1) = sinf((*this)(2))*sinf((*this)(1))*sinf((*this)(0))+cosf((*this)(2))*cosf((*this)(0));
    mResult(1,2) = cosf((*this)(1))*sinf((*this)(0));
    mResult(2,0) = cosf((*this)(2))*sinf((*this)(1))*cosf((*this)(0))+sinf((*this)(2))*sinf((*this)(0));
    mResult(2,1) = sinf((*this)(2))*sinf((*this)(1))*cosf((*this)(0))-cosf((*this)(2))*sinf((*this)(0));
    mResult(2,2) = cosf((*this)(1))*cosf((*this)(0));

    return mResult;
}
Quaternion Euler::toQtn() const
{
    return (*this).toDCM().toQtn();
}
Euler operator*(const Euler& c_eulL,const Euler& c_eulR)
{
    return (c_eulL.toDCM()*c_eulR.toDCM()).toEul();
}
Vec operator*(const Euler& c_eulValue,const Vec& c_vValue)
{
    return c_eulValue.toDCM()*c_vValue;
}

// --------------------------
// Quaternion
// --------------------------
Quaternion::Quaternion()
    :Vec(4)
{

}
Quaternion::Quaternion(const Mat& c_mValue)
    :Vec(c_mValue)
{
    if( c_mValue._nRows!=4 )
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
    Quaternion qtnResult;

    qtnResult(0) =-(*this)(0);
    qtnResult(1) =-(*this)(1);
    qtnResult(2) =-(*this)(2);
    qtnResult(3) = (*this)(3);

    return qtnResult;
}
Vec Quaternion::axis() const
{
    Vec vResult(3);

    vResult(0) =(*this)(0);
    vResult(1) =(*this)(1);
    vResult(2) =(*this)(2);

    return vResult;
}
Mat Quaternion::dot_E() const
{
    Mat mResult(4,3);

    mResult(0,0) = (*this)(3);
    mResult(0,1) =-(*this)(2);
    mResult(0,2) = (*this)(1);
    mResult(1,0) = (*this)(2);
    mResult(1,1) = (*this)(3);
    mResult(1,2) =-(*this)(0);
    mResult(2,0) =-(*this)(1);
    mResult(2,1) = (*this)(0);
    mResult(2,2) = (*this)(3);
    mResult(3,0) = (*this)(0);
    mResult(3,1) =-(*this)(1);
    mResult(3,2) =-(*this)(2);

    return mResult;
}
Quaternion Quaternion::dot(const Vec& c_vRps) const
{
    Quaternion qtnResult;

    Mat mEquation =(*this).dot_E();

    qtnResult(0) =mEquation(0,0)*c_vRps(0)+mEquation(0,1)*c_vRps(1)+mEquation(0,2)*c_vRps(2);
    qtnResult(1) =mEquation(1,0)*c_vRps(0)+mEquation(1,1)*c_vRps(1)+mEquation(1,2)*c_vRps(2);
    qtnResult(2) =mEquation(2,0)*c_vRps(0)+mEquation(2,1)*c_vRps(1)+mEquation(2,2)*c_vRps(2);
    qtnResult(3) =mEquation(3,0)*c_vRps(0)+mEquation(3,1)*c_vRps(1)+mEquation(3,2)*c_vRps(2);

    return qtnResult;
}
void Quaternion::normalized()
{
    float fNorm =(*this).norm();

    if( fNorm<=0.0f )
    {
        throw std::runtime_error(
            "Quaternion::normalized: zero norm"
        );
    }
    //
    (*this)(0) /=fNorm;
    (*this)(1) /=fNorm;
    (*this)(2) /=fNorm;
    (*this)(3) /=fNorm;
}
Mat Quaternion::toDCM() const
{
    Mat mResult(3,3);

    mResult(0,0) =1.0-2.0*((*this)(1)*(*this)(1)+(*this)(2)*(*this)(2));
    mResult(0,1) =2.0*((*this)(0)*(*this)(1)+(*this)(2)*(*this)(3));
    mResult(0,2) =2.0*((*this)(2)*(*this)(0)-(*this)(1)*(*this)(3));
    mResult(1,0) =2.0*((*this)(0)*(*this)(1)-(*this)(2)*(*this)(3));
    mResult(1,1) =1.0-2.0*((*this)(0)*(*this)(0)+(*this)(2)*(*this)(2));
    mResult(1,2) =2.0*((*this)(1)*(*this)(2)+(*this)(0)*(*this)(3));
    mResult(2,0) =2.0*((*this)(2)*(*this)(0)+(*this)(1)*(*this)(3));
    mResult(2,1) =2.0*((*this)(1)*(*this)(2)-(*this)(0)*(*this)(3));
    mResult(2,2) =1.0-2.0*((*this)(0)*(*this)(0)+(*this)(1)*(*this)(1));

    return mResult;
}
Euler Quaternion::toEul() const
{
    return (*this).toDCM().toEul();
}
Quaternion operator*(const Quaternion& c_qtnL,const Quaternion& c_qtnR)
{
    Quaternion qtnResult;

    Vec vL =c_qtnL.axis();
    Vec vR =c_qtnR.axis();
    Vec vRightLeft =vL*c_qtnR(3);
    Vec vLeftRight =vR*c_qtnL(3);
    Vec vCross =vR.cpx( vL );
    float fDot =vR.dot( vL );

    qtnResult(0) =vRightLeft(0)+vLeftRight(0)+vCross(0);
    qtnResult(1) =vRightLeft(1)+vLeftRight(1)+vCross(1);
    qtnResult(2) =vRightLeft(2)+vLeftRight(2)+vCross(2);
    qtnResult(3) =c_qtnR(3)*c_qtnL(3)-fDot;
    qtnResult.normalized();

    return qtnResult;
}
Vec operator*(const Quaternion& c_qtnValue,const Vec& c_vValue)
{
    return c_qtnValue.toDCM()*c_vValue;
}
