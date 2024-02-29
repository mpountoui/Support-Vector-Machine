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
    SVM(SVM const&);
    SVM(SVM&&);
    SVM& operator=(SVM const&);
    SVM& operator=(SVM&&);
    ~SVM();

private:
    double operator()(Vector const&) const;

public:
    std::vector<double> const& LagrangeMultipliers() const;
    int Predict(Vector const&) const;
    void Fit(std::vector<Vector> const&, std::vector<int> const&);
    std::vector<int> Predict(std::vector<Vector> const&) const;
};

/* ----------------------------------------------------------------------------------------- */

struct DataSet
{
    int label_1 = 0;
    int label_2 = 0;
    std::vector<Vector> x;
    std::vector<int>    y;
};

/* ----------------------------------------------------------------------------------------- */

class MultiClassSVM
{
    KernelType m_kernel_type;
    size_t m_degree;
    double m_gamma;
    double m_Coef;
    double m_C;
    std::vector<std::tuple<SVM, int, int>> m_Models;

public:
    MultiClassSVM(KernelType, double, size_t, double, double);
    MultiClassSVM(MultiClassSVM const&)            = delete;
    MultiClassSVM(MultiClassSVM&&)                 = delete;
    MultiClassSVM& operator=(MultiClassSVM const&) = delete;
    MultiClassSVM& operator=(MultiClassSVM&&)      = delete;
    ~MultiClassSVM()                               = default;

private:
    std::vector<DataSet> PrepareData(std::vector<Vector> const&, std::vector<int> const&) const;

public:
    std::vector<double> const LagrangeMultipliers() const;
    void Fit(std::vector<Vector> const&, std::vector<int> const&);
    std::vector<int> Predict(std::vector<Vector> const&) const;
};

#endif
