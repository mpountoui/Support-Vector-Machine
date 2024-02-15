#include <vector>
#include <cassert>
#include "Support_vector_machine.hpp"
#include "Sequential_minimal_optimization.hpp"

/* ----------------------------------------------------------------------------------------- */

SVM::SVM(KernelType kernel_type, double C, size_t degree, double gamma, double Coeff)
:
m_C(C)
{
    m_Kernel = KernelFactory(kernel_type, degree, gamma, Coeff);
}

/* ----------------------------------------------------------------------------------------- */

SVM::~SVM()
{
    delete m_Kernel;
}

/* ----------------------------------------------------------------------------------------- */

std::vector<double> const& SVM::LagrangeMultipliers() const
{
    return m_a;
}

/* ----------------------------------------------------------------------------------------- */

double SVM::operator()(Vector const& X) const
{
    double ret = 0.0;

    for(size_t i = 0; i < m_a.size(); ++i)
    {
        ret += m_y->operator[](i) * m_a[i] * m_Kernel->operator()( m_x->operator[](i), X );
    }

    return ret - m_b;
}

/* ----------------------------------------------------------------------------------------- */

void SVM::Fit(std::vector<Vector> const& x, std::vector<int> const& y)
{
    if( x.size() != y.size() ){ assert(false); return; }

    m_x = &x;
    m_y = &y;

    m_a = std::vector<double>(x.size(), 0.5);
    m_Kernel->W_InitializeIfLinearKernel(y, m_a, x);
    DualProblem(*this).Solve();
    m_Kernel->SaveSupportVectorsIfNonLinear(y, m_a, x);
}

/* ----------------------------------------------------------------------------------------- */

std::vector<int> SVM::Predict(std::vector<Vector> const& x_vec) const
{
    std::vector<int> y_vec;
    y_vec.reserve(x_vec.size());

    for(auto const& x : x_vec)
    {
        double ret = m_Kernel->Predict(x, m_b);

        if( ret > 0 )
        {
            y_vec.push_back(1);
        }
        else
        {
            y_vec.push_back(-1);
        }
    }
    return y_vec;
}
