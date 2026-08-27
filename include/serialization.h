#pragma once
#include "cuda_matrix.h"
#include "matrix.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
namespace serialization {
    struct MatrixData
    {
        std::uint32_t rows, cols;
        std::vector<float> values;
    };
    inline void write_exact(FILE *file,const void *p,std::size_t n)
    {
        if( fwrite(p,1,n,file)!=n )
        {
            throw std::runtime_error(
                "serialization: write failed"
            );
        }
    }
    inline void read_exact(FILE *file, void *p, std::size_t n) 
    {
        if( fread(p,1,n,file)!=n )
        {
            throw std::runtime_error(
                "serialization: truncated file"
            );
        }
    }
    template <class T> void write(FILE *file,const T &v)
    {
        write_exact( file,&v,sizeof(v) );
    }
    template <class T> T read(FILE *file)
    {
        T v{};
        read_exact( file,&v,sizeof(v) );

        return v;
    }
    inline MatrixData host(const cuMat &m)
    {
        if( (m._nRows<=0)||(m._nCols<=0) )
        {
            throw std::runtime_error(
                "serialization: invalid shape"
            );
        }
        MatrixData d{
            static_cast<std::uint32_t>(m._nRows),
            static_cast<std::uint32_t>(m._nCols),
            {}
        };
        d.values.resize( static_cast<std::size_t>(d.rows)*d.cols );
        Mat h( m._nRows,m._nCols ); m.upload(h);
        std::copy( h._lpfHost,h._lpfHost+d.values.size(),d.values.begin() );

        return d;
    }
    inline void write_matrix(FILE *file,const cuMat &m)
    {
        auto d = host(m);
        write(file, d.rows);
        write(file, d.cols);
        write_exact(file, d.values.data(), d.values.size() * sizeof(float));
    }
    inline MatrixData read_matrix(FILE *file)
    {
        MatrixData d{read<std::uint32_t>(file), read<std::uint32_t>(file), {}};
        if (!d.rows || !d.cols ||
            static_cast<std::uint64_t>(d.rows) * d.cols > 1000000000ULL)
        {
            throw std::runtime_error("serialization: invalid shape");
        }
        d.values.resize(static_cast<std::size_t>(d.rows) * d.cols);
        read_exact(file, d.values.data(), d.values.size() * sizeof(float));

        return d;
    }
    inline void validate(const MatrixData &d,const cuMat &m)
    {
        if (d.rows != static_cast<std::uint32_t>(m._nRows) ||
            d.cols != static_cast<std::uint32_t>(m._nCols))
        {
            throw std::runtime_error("serialization: shape mismatch");
        }
    }
    inline void apply(const MatrixData &d,cuMat &m)
    {
        validate(d, m);
        Mat h(m._nRows, m._nCols);
        std::copy(d.values.begin(), d.values.end(), h._lpfHost);
        m.download(h);
    }
    inline void eof(FILE *file)
    {
        char c;
        if (std::fread(&c, 1, 1, file) == 1) {
            throw std::runtime_error("serialization: trailing data");
        }
    }
} // namespace serialization
