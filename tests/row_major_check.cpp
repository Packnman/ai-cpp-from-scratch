#include "cuda_matrix.h"
#include "cuda_tensor.h"
#include "matrix.h"
#include "module.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition,const char* message)
{
    if(!condition) throw std::runtime_error(message);
}
void near(float actual,float expected,const char* message)
{
    if(std::fabs(actual-expected)>1.0e-4f) throw std::runtime_error(message);
}
void put(cufMat& tensor,const std::vector<float>& values)
{
    tensor.copyFromHost(values.data(),values.size());
}

void matrixChecks()
{
    Mat a(2,3);
    for(int row=0;row<2;++row) for(int col=0;col<3;++col) a(row,col)=float(row*3+col+1);
    require(a._lpfHost[0]==1 && a._lpfHost[1]==2 && a._lpfHost[2]==3 &&
            a._lpfHost[3]==4 && a._lpfHost[4]==5 && a._lpfHost[5]==6,
            "Mat is not row-major");
    Mat t=a.trp();
    require(t.rows()==3 && t.cols()==2 && t(2,1)==6,"Mat transpose failed");
    Mat b(3,2);
    b(0,0)=7; b(0,1)=8; b(1,0)=9; b(1,1)=10; b(2,0)=11; b(2,1)=12;
    Mat c=a*b;
    near(c(0,0),58,"Mat multiply failed"); near(c(1,1),154,"Mat multiply failed");
}

std::vector<float> transposeHost(const std::vector<float>& input,int rows,int cols)
{
    std::vector<float> result(cols*rows);
    for(int r=0;r<rows;++r) for(int c=0;c<cols;++c) result[c*rows+r]=input[r*cols+c];
    return result;
}
void gemmChecks()
{
    const std::vector<float> effectiveA{1,2,3,4,5,6};
    const std::vector<float> effectiveB{1,2,3,4,5,6,7,8,9,10,11,12};
    const std::vector<float> expected{38,44,50,56,83,98,113,128};
    for(bool ta:{false,true}) for(bool tb:{false,true}) {
        cufMat a(ta?3:2,ta?2:3),b(tb?4:3,tb?3:4),c(2,4);
        put(a,ta?transposeHost(effectiveA,2,3):effectiveA);
        put(b,tb?transposeHost(effectiveB,3,4):effectiveB);
        cuda_fill(c,0); cuda_gemm(c,a,b,ta,tb);
        auto actual=c.toHost();
        for(std::size_t i=0;i<expected.size();++i) near(actual[i],expected[i],"cuda_gemm transpose combination failed");
    }
    cufMat a(2,3),t(3,2); put(a,effectiveA); cuda_transpose(t,a);
    require(t.toHost()==transposeHost(effectiveA,2,3),"cuda_transpose failed");
}

cufMat survivingView()
{
    cufMat source({2,3,4});
    std::vector<float> values(24); for(int i=0;i<24;++i) values[i]=float(i);
    put(source,values);
    return source.slice(1,1,3);
}
void tensorChecks()
{
    cufMat tensor({2,3,4});
    require(tensor.shape()==std::vector<std::int64_t>({2,3,4}),"shape failed");
    require(tensor.strides()==std::vector<std::int64_t>({12,4,1}),"strides failed");
    require(tensor.numel()==24 && tensor.isContiguous(),"numel/contiguous failed");
    std::vector<float> values(24); for(int i=0;i<24;++i) values[i]=float(i);
    put(tensor,values);
    auto reshaped=tensor.reshape({4,6});
    require(reshaped.data()==tensor.data() && reshaped.strides()==std::vector<std::int64_t>({6,1}),"reshape view failed");
    auto permuted=tensor.permute({1,0,2});
    require(!permuted.isContiguous() && permuted.shape()==std::vector<std::int64_t>({3,2,4}),"permute metadata failed");
    const std::vector<float> expectedPermuted{0,1,2,3,12,13,14,15,4,5,6,7,16,17,18,19,8,9,10,11,20,21,22,23};
    auto packed=permuted.contiguous();
    require(packed.isContiguous() && packed.toHost()==expectedPermuted,"permute contiguous failed");
    const std::vector<float> expectedSlice{4,5,6,7,8,9,10,11,16,17,18,19,20,21,22,23};
    require(survivingView().contiguous().toHost()==expectedSlice,"slice or storage lifetime failed");
}

class StateModel final: public Model
{
public:
    Tensor value{{2,3,4}};
    StateModel() { registerParameter("value",&value); }
    std::shared_ptr<Tensor> forward(std::vector<std::shared_ptr<Tensor>>& inputs) override
    { return inputs.empty()?nullptr:inputs[0]; }
};
void checkpointChecks()
{
    const char* file="row_major_nd_checkpoint.bin";
    StateModel source,destination;
    std::vector<float> values(24); for(int i=0;i<24;++i) values[i]=float(i*i);
    put(source.value._mData,values); source.save(file); destination.load(file);
    require(destination.value._mData.shape()==source.value._mData.shape() &&
            destination.value._mData.toHost()==values,"N-D checkpoint failed");
    std::remove(file);
}
}

int main()
{
    try { matrixChecks(); gemmChecks(); tensorChecks(); checkpointChecks(); std::cout<<"row-major check passed\n"; return 0; }
    catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
