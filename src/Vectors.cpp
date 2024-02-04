#include <vector>
#include "Vectors.hpp"
#include <cassert>

/* ----------------------------------------------------------------------------------------- */

double operator*(Vector const& v1, Vector const& v2)
{
    if( v1.size() != v2.size() ){ assert(false); return 0.0; }

    double DotProduct = 0.0;

    for(size_t q = 0; q < v1.size(); ++q)
    {
        DotProduct += v1[q] * v2[q];
    }

    return DotProduct;
}

/* ----------------------------------------------------------------------------------------- */

Vector& operator+=(Vector& lhs, Vector const& rhs)
{
    if( lhs.size() != rhs.size() ){ assert(false); return lhs; }

    for(size_t q = 0; q < lhs.size(); ++q)
    {
        lhs[q] += rhs[q];
    }

    return lhs;
}

/* ----------------------------------------------------------------------------------------- */

Vector& operator-=(Vector& lhs, Vector const& rhs)
{
    if( lhs.size() != rhs.size() ){ assert(false); return lhs; }

    for(size_t q = 0; q < lhs.size(); ++q)
    {
        lhs[q] -= rhs[q];
    }

    return lhs;
}

/* ----------------------------------------------------------------------------------------- */

Vector operator+(Vector const& v1, Vector const& v2)
{
    if( v1.size() != v2.size() ){ assert(false); return Vector(); }

    Vector sum;
    sum.reserve(v1.size());

    for(size_t q = 0; q < v1.size(); ++q)
    {
        sum.push_back( v1[q] + v2[q] );
    }

    return sum;
}

/* ----------------------------------------------------------------------------------------- */

Vector operator-(Vector const& v1, Vector const& v2)
{
    if( v1.size() != v2.size() ){ assert(false); return Vector(); }

    Vector diff;
    diff.reserve(v1.size());

    for(size_t q = 0; q < v1.size(); ++q)
    {
        diff.push_back( v1[q] - v2[q] );
    }

    return diff;
}
