#pragma once

#define IDX2F(row,col,rows)   ((col)*(rows)+(row))      // column-major(Fortran)
#define IDX2C(row,col,cols)   ((row)*(cols)+(col))      // row-major(C/C++)

class Mat;

class cuMat{
public:
    cuMat(int nRows,int nCols);
    cuMat(const cuMat& val);
    ~cuMat();
private:
public:
    float*  _lpfDevice;
    int     _nRows;
    int     _nCols;
public:
    void upload(Mat& host);
    void download(const Mat& host);
    void ones();
    
public:
    cuMat& operator=(const cuMat& val);         // コピー演算子
    cuMat& operator=(cuMat&& val) noexcept;     // ムーブ演算子
};

void cuda_axpy(cuMat& rst,float alpha,const cuMat& A);  // R += αA
void cuda_geam(cuMat& rst,float alpha,const cuMat& A,float beta,const cuMat& B);  // R = αA + βB
// R = α*op(A)*op(B) + β*op(B)
void cuda_gemm(cuMat& rst,const cuMat& A,const cuMat& B,
               bool trpA=false,
               bool trpB=false,
               float alpha=1.0f,
               float beta=0.0f
);
void cuda_fill(cuMat& rst,float value); // R[:] = value
void cuda_transpose(cuMat& rst,const cuMat& A); // R = A^T
void cuda_scale(cuMat& rst,float value);    // R *= value
void cuda_mul_elementwise(cuMat& rst,const cuMat& A,const cuMat& B);  // R = A ⦿ B
void cuda_ReLU_forward(cuMat& rst,const cuMat& value);
void cuda_ReLU_backward(
    cuMat& rst,         // 求める勾配 ∂L/∂x を加算する先
    const cuMat& data,  // ReLUへの入力 x
    const cuMat& grad   // 上流から来た勾配 ∂L/∂y
);