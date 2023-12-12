#include <vector>
#include <cassert>
#include "Support_vector_machine.hpp"

/* ----------------------------------------------------------------------------------------- */

double SVM::operator()(std::vector<double> const& X) const
{
    double ret = 0.0;

    for(size_t i = 0; i < m_LagrMult.size(); ++i)
    {
        ret += m_Targets[i] * m_LagrMult[i] * m_Kernel(m_Samples[i], X);
    }

    return ret - m_b;
}

/* ----------------------------------------------------------------------------------------- */

void SVM::fit(std::vector<std::vector<double>> const& x, std::vector<int> const& y) const
{
    if( x.size() != y.size() ){ assert(false); return; }
    for(size_t i = 0; i < x.size(); ++i)
    {
        for(auto feature : x[i])
        {
            printf("%lf   " , feature);
        }
        printf("\ny %d\n", y[i]);
    }
}

/* ----------------------------------------------------------------------------------------- */

void SVM::Test()
{
    printf("Hellooooooo!!!\n");
}
