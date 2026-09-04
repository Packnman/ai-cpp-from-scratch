#pragma once

#define IDX2F(nRow,nCol,nRows)   ((nCol)*(nRows)+(nRow))      // column-major(Fortran)
#define IDX2C(nRow,nCol,nCols)   ((nRow)*(nCols)+(nCol))      // row-major(C/C++)

class Euler;
class Quaternion;


// --------------------------
// Matrix
// --------------------------
class Mat{
public:
    Mat(int nRows,int nCols);
    Mat(const Mat& c_mValue);
    Mat(Mat&& mValue) noexcept;
    ~Mat();
private:
public:
    int     _nRows;
    int     _nCols;
    float*  _lpfHost;
public:
    void ones();
    float tri() const;
    Mat trp() const;
    virtual Euler toEul() const;
    virtual Quaternion toQtn() const;
public:
    Mat& operator=(const Mat& c_mValue);         // コピー演算子
    Mat& operator=(Mat&& mValue) noexcept;       // ムーブ演算子
    Mat& operator+=(const Mat& c_mValue);
    Mat& operator-=(const Mat& c_mValue);
    Mat& operator*=(const Mat& c_mValue);
    Mat& operator*=(float fValue);
    float& operator()(int nRow,int nCol);
    const float& operator()(int nRow,int nCol) const;
};
Mat operator+(const Mat& c_mL,const Mat& c_mR);
Mat operator-(const Mat& c_mL,const Mat& c_mR);
Mat operator*(const Mat& c_mL,const Mat& c_mR);

// --------------------------
// Vector
// --------------------------
class Vec : public Mat{
public:
    Vec(int nRows);
    Vec(const Mat& c_mValue);
    ~Vec();
public:
    Mat skew() const;
    Vec cpx(const Vec& c_vValue) const;
    float dot(const Vec& c_vValue) const;
    float norm() const;
    Quaternion exp(float fDt) const;
public:
    float& operator()(int nRow);
    const float& operator()(int nRow) const;
};
Vec operator*(const Mat& c_mL,const Vec& c_vR);

