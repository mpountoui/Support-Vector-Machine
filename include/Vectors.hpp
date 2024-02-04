#ifndef VECTORS
#define VECTORS

#include <vector>

/* ----------------------------------------------------------------------------------------- */

using Vector = std::vector<double>;

Vector& operator+=(Vector&      , Vector const&);
Vector& operator-=(Vector&      , Vector const&);
Vector  operator+ (Vector const&, Vector const&);
Vector  operator- (Vector const&, Vector const&);
double  operator* (Vector const&, Vector const&);

/* ----------------------------------------------------------------------------------------- */

template<typename T> Vector operator*(T factor, Vector const& v)
{
    Vector fv;
    fv.reserve(v.size());

    for(size_t q = 0; q < v.size(); ++q)
    {
        fv.push_back( factor * v[q] );
    }

    return fv;
}

#endif