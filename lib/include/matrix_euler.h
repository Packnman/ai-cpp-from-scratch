#pragma once

#include "matrix.h"

// --------------------------
// Euler
// --------------------------
class Euler : public Vec{
public:
    Euler();
    Euler(const Mat& c_mValue);
    ~Euler();
public:
    Mat toDCM() const;
    virtual Quaternion toQtn() const;
public:
    friend Euler operator*(const Euler& c_eulL,const Euler& c_eulR);
    friend Vec operator*(const Euler& c_eulValue,const Vec& c_vValue);
};