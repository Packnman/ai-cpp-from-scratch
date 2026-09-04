#include <iostream>
#include <math.h>
#include "matrix_euler.h"
#include "matrix_quaternion.h"


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
