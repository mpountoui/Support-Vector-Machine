#ifndef SUPPORT_VECTORS_MACHINE
#define SUPPORT_VECTORS_MACHINE

#include <vector>
#include "Kernels.hpp"

/* ----------------------------------------------------------------------------------------- */

class SVM
{
    friend class DualProblem;

private:
    double m_b = 0.0;
    double m_C = 10.0;
    std::vector<double> m_a; /* Lagrange Multipliers */
    std::vector<Vector> const* m_x = nullptr; /* Samples */
    std::vector<int>    const* m_y = nullptr; /* Targets */
    Kernel* m_Kernel = nullptr;

public:
    SVM(KernelType, double, size_t, double, double);
    SVM(SVM const&)            = delete;
    SVM(SVM&&)                 = delete;
    SVM& operator=(SVM const&) = delete;
    SVM& operator=(SVM&&)      = delete;
    ~SVM();

private:
    double operator()(Vector const&) const;

public:
    std::vector<double> const& LagrangeMultipliers() const;
    void Fit(std::vector<Vector> const&, std::vector<int> const&);
    std::vector<int> Predict(std::vector<Vector> const&) const;
};

#endif
