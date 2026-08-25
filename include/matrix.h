#pragma once

#define IDX2F(row,col,rows)   ((col)*(rows)+(row))      // column-major(Fortran)
#define IDX2C(row,col,cols)   ((row)*(cols)+(col))      // row-major(C/C++)

class Euler;
class Quaternion;


// --------------------------
// Matrix
// --------------------------
class Mat{
public:
    Mat(int nRows,int nCols);
    Mat(const Mat& val);
    Mat(Mat&& val) noexcept;
    ~Mat();
private:
public:
    float*  _lpfHost;
    int     _nRows;
    int     _nCols;
public:
    void ones();
    float tri() const;
    Mat trp() const;
    virtual Euler toEul() const;
    virtual Quaternion toQtn() const;
public:
    Mat& operator=(const Mat& val);         // コピー演算子
    Mat& operator=(Mat&& val) noexcept;     // ムーブ演算子
    Mat& operator+=(const Mat& val);
    Mat& operator-=(const Mat& val);
    Mat& operator*=(const Mat& val);
    Mat& operator*=(float val);
    float& operator()(int row,int col);
    const float& operator()(int row,int col) const;
};
Mat operator+(const Mat& L,const Mat& R);
Mat operator-(const Mat& L,const Mat& R);
Mat operator*(const Mat& L,const Mat& R);

// --------------------------
// Vector
// --------------------------
class Vec : public Mat{
public:
    Vec(int nRows);
    Vec(const Mat& val);
    ~Vec();
public:
    Mat skew() const;
    Vec cpx(const Vec& val) const;
    float dot(const Vec& val) const;
    float norm() const;
    Quaternion exp(float dt) const;
public:
    float& operator()(int row);
    const float& operator()(int row) const;
};
Vec operator*(const Mat& L,const Vec& R);

// --------------------------
// Euler
// --------------------------
class Euler : public Vec{
public:
    Euler();
    Euler(const Mat& val);
    ~Euler();
public:
    Mat toDCM() const;
    virtual Quaternion toQtn() const;
public:
    friend Euler operator*(const Euler& e1,const Euler& e2);
    friend Vec operator*(const Euler& e,const Vec& v);
};

// --------------------------
// Quaternion
// --------------------------
class Quaternion : public Vec{
public:
    Quaternion();
    Quaternion(const Mat& val);
    ~Quaternion();
public:
    Quaternion cnj() const;
    Vec axis() const;
    Mat dot_E() const;
    Quaternion dot(const Vec& rps) const;
    void normalized();
    //
    Mat toDCM() const;
    virtual Euler toEul() const;
public:
    friend Quaternion operator*(const Quaternion& q1,const Quaternion& q2);
    friend Vec operator*(const Quaternion& q,const Vec& v);
};