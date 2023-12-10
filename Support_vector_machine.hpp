#include <vector>
#include "Kernels.hpp"

class SVM
{
    friend class QuadraticProgrammingProblem;

private:
    double m_b;
    std::vector<double> m_LagrMult; /* Lagrange Multipliers */
    std::vector<double> m_Targets;
    std::vector<std::vector<double>> m_Samples;
    Kernel m_Kernel;

public:
    SVM()                      = default;
    SVM(SVM const&)            = delete;
    SVM(SVM&&)                 = delete;
    SVM& operator=(SVM const&) = delete;
    SVM& operator=(SVM&&)      = delete;
    ~SVM()                     = default;

public:
    double operator()(std::vector<double> const&) const;
};
