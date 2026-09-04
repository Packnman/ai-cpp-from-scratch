#include <iostream>
#include <math.h>
#include "matrix_quaternion.h"
#include "matrix_euler.h"


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
