#pragma once

#define IDX2F(row,col,rows)   ((col)*(rows)+(row))      // column-major(Fortran)
#define IDX2C(row,col,cols)   ((row)*(cols)+(col))      // row-major(C/C++)

class cuMat{
public:
    cuMat(int nRows,int nCols);
    cuMat(const cuMat& val);
    ~cuMat();
private:
    float   *_lpfDevice;
    float   *_lpfHost;
    int     _nRows;
    int     _nCols;
public:
    void upload();
    void download();
public:
    cuMat& operator=(const cuMat& val);         // コピー演算子
    cuMat& operator=(cuMat&& val) noexcept;     // ムーブ演算子
    friend cuMat operator+(const cuMat& L,const cuMat& R);
    friend cuMat operator-(const cuMat& L,const cuMat& R);
    friend cuMat operator*(const cuMat& L,const cuMat& R);
    cuMat& operator+=(const cuMat& val);
    cuMat& operator-=(const cuMat& val);
    cuMat& operator*=(const cuMat& val);
    cuMat& operator*=(float val);
    float& operator()(int row,int col);
    const float& operator()(int row,int col) const;
};

