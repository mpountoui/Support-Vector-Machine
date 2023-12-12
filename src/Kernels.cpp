#include <vector>
#include "Kernels.hpp"

double Kernel::operator()(std::vector<double> const& X_1, std::vector<double> const& X_2) const
{
    double ret = 0.0;
    
    for(size_t i = 0; i < X_1.size(); ++i)
    {
        ret += X_1[i] * X_2[i];
    }

    return ret;
}
