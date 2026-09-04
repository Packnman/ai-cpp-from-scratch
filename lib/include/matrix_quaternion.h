#pragma once

#include "matrix.h"


// --------------------------
// Quaternion
// --------------------------
class Quaternion : public Vec{
public:
    Quaternion();
    Quaternion(const Mat& c_mValue);
    ~Quaternion();
public:
    Quaternion cnj() const;
    Vec axis() const;
    Mat dot_E() const;
    Quaternion dot(const Vec& c_vRps) const;
    void normalized();
    //
    Mat toDCM() const;
    virtual Euler toEul() const;
public:
    friend Quaternion operator*(
        const Quaternion& c_qtnL,
        const Quaternion& c_qtnR
    );
    friend Vec operator*(const Quaternion& c_qtnValue,const Vec& c_vValue);
};
