#pragma once

#define IDX2F(row,col,rows)   ((col)*(rows)+(row))      // column-major(Fortran)
#define IDX2C(row,col,cols)   ((row)*(cols)+(col))      // row-major(C/C++)

class Mat{
public:
    Mat(int nRows,int nCols);
    Mat(const Mat& val);
    ~Mat();
private:
public:
    float*  _lpfHost;
    int     _nRows;
    int     _nCols;
public:
    void ones();
    Mat transpose();
public:
    Mat& operator=(const Mat& val);         // コピー演算子
    Mat& operator=(Mat&& val) noexcept;     // ムーブ演算子
    friend Mat operator+(const Mat& L,const Mat& R);
    friend Mat operator-(const Mat& L,const Mat& R);
    friend Mat operator*(const Mat& L,const Mat& R);
    Mat& operator+=(const Mat& val);
    Mat& operator-=(const Mat& val);
    Mat& operator*=(const Mat& val);
    Mat& operator*=(float val);
    float& operator()(int row,int col);
    const float& operator()(int row,int col) const;
};

