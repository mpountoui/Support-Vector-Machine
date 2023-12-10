#include <vector>
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