#include <vector>
#include <cassert>
#include <cmath>
#include "Kernels.hpp"
#include "Vectors.hpp"

/* ----------------------------------------------------------------------------------------- */

class LinearKernel final : private Kernel
{
private:
    friend Kernel* ContructKernel(KernelType);

private:
    Vector m_W;

private:
    LinearKernel()                               = default;
    LinearKernel(LinearKernel const&)            = delete;
    LinearKernel(LinearKernel&&)                 = delete;
    LinearKernel& operator=(LinearKernel const&) = delete;
    LinearKernel& operator=(LinearKernel&&)      = delete;

public:
    ~LinearKernel()                              = default;

public:
    void W_InitializeIfLinearKernel(std::vector<int>    const&,
                                    std::vector<double> const&,
                                    std::vector<Vector> const&) override;

    void W_UpdateIfLinearKernel(int, double, Vector const&,
                                int, double, Vector const&) override;
    
    void SaveSupportVectorsIfNonLinear(std::vector<int>    const&,
                                       std::vector<double> const&,
                                       std::vector<Vector> const&) override;

    double Predict(Vector const&, double) override;
        
    double operator()(Vector const&, Vector const&) const override;
};

/* ----------------------------------------------------------------------------------------- */

class NonLinearKernel : protected Kernel
{
protected:
    std::vector<SupportVectors> m_SupportVectors;

protected:
    NonLinearKernel()                                  = default;
    NonLinearKernel(NonLinearKernel const&)            = delete;
    NonLinearKernel(NonLinearKernel&&)                 = delete;
    NonLinearKernel& operator=(NonLinearKernel const&) = delete;
    NonLinearKernel& operator=(NonLinearKernel&&)      = delete;

public:
    ~NonLinearKernel() override                        = default;

 public:    
    void W_InitializeIfLinearKernel(std::vector<int>    const&,
                                    std::vector<double> const&,
                                    std::vector<Vector> const&) override;

    void W_UpdateIfLinearKernel(int, double, Vector const&,
                                int, double, Vector const&) override;
    
    void SaveSupportVectorsIfNonLinear(std::vector<int>    const&,
                                       std::vector<double> const&,
                                       std::vector<Vector> const&) override;

    double Predict(Vector const&, double) override;
};

/* ----------------------------------------------------------------------------------------- */

class PolynomialKernel final : private NonLinearKernel
{
private:
    friend Kernel* ContructKernel(KernelType);

private:
    PolynomialKernel()                                   = default;
    PolynomialKernel(PolynomialKernel const&)            = delete;
    PolynomialKernel(PolynomialKernel&&)                 = delete;
    PolynomialKernel& operator=(PolynomialKernel const&) = delete;
    PolynomialKernel& operator=(PolynomialKernel&&)      = delete;

public:
    ~PolynomialKernel()                                  = default;

public:
    double operator()(Vector const&, Vector const&) const;
};

/* ----------------------------------------------------------------------------------------- */

class RbfKernel final : private NonLinearKernel
{
private:
    friend Kernel* ContructKernel(KernelType);

private:
    double m_gamma = 2.0;

private:
    RbfKernel()                            = default;
    RbfKernel(RbfKernel const&)            = delete;
    RbfKernel(RbfKernel&&)                 = delete;
    RbfKernel& operator=(RbfKernel const&) = delete;
    RbfKernel& operator=(RbfKernel&&)      = delete;

public:
    ~RbfKernel()                           = default;

public:
    double operator()(Vector const&, Vector const&) const;
};

/* ----------------------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------------------- */
/* ----------------------------------- LinearKernel ---------------------------------------- */
/* ----------------------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------------------- */

void LinearKernel::W_InitializeIfLinearKernel(std::vector<int> const& y, std::vector<double> const& a, std::vector<Vector> const& x)
{
    m_W = std::vector<double>(x[0].size(), 0.0);

    for(int i = 0; i < a.size(); ++i)
    {
        m_W += y[i] * a[i] * x[i];
    }
}

/* ----------------------------------------------------------------------------------------- */

void LinearKernel::W_UpdateIfLinearKernel(int y_1, double da_1, Vector const& x_1, 
                                          int y_2, double da_2, Vector const& x_2)
{
    m_W += y_1 * da_1 * x_1 + y_2 * da_2 * x_2;
}

/* ----------------------------------------------------------------------------------------- */

void LinearKernel::SaveSupportVectorsIfNonLinear(std::vector<int> const& y, std::vector<double> const& a, std::vector<Vector> const& x)
{}

/* ----------------------------------------------------------------------------------------- */

double LinearKernel::Predict(Vector const& x, double b)
{
    return m_W * x - b;
}

/* ----------------------------------------------------------------------------------------- */

double LinearKernel::operator()(Vector const& x_1, Vector const& x_2) const
{
    return x_1 * x_2;
}

/* ----------------------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------------------- */
/* ---------------------------------- NonLinearKernel -------------------------------------- */
/* ----------------------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------------------- */

void NonLinearKernel::W_InitializeIfLinearKernel(std::vector<int> const& y, std::vector<double> const& a, std::vector<Vector> const& x)
{}

/* ----------------------------------------------------------------------------------------- */

void NonLinearKernel::W_UpdateIfLinearKernel(int, double, Vector const&, int, double, Vector const&)
{}

/* ----------------------------------------------------------------------------------------- */

void NonLinearKernel::SaveSupportVectorsIfNonLinear(std::vector<int> const& y, std::vector<double> const& a, std::vector<Vector> const& x)
{
    m_SupportVectors.reserve( a.size() / 10 );

    for(size_t i = 0; i < a.size(); ++i)
    {
        if( a[i] > 1e-14 )
        {
            m_SupportVectors.push_back( {y[i], a[i], x[i]} );
        }
    }
}

/* ----------------------------------------------------------------------------------------- */

double NonLinearKernel::Predict(Vector const& x, double b)
{
    double ret = 0.0;

    for(auto const SupVec : m_SupportVectors)
    {
        ret += SupVec.y * SupVec.a * this->operator()(SupVec.x , x);
    }

    return ret - b;
}

/* ----------------------------------------------------------------------------------------- */

double PolynomialKernel::operator()(Vector const& x_1, Vector const& x_2) const
{
    assert(false);
    return 0.0;
}

/* ----------------------------------------------------------------------------------------- */

double RbfKernel::operator()(Vector const& x_1, Vector const& x_2) const
{
    return std::exp( - m_gamma * (x_1-x_2)*(x_1-x_2) );
}

/* ----------------------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------------------- */

Kernel* ContructKernel(KernelType type)
{
    switch (type)
    {
    case LINEAR:
        return new LinearKernel();
    case POLYNOMIAL:
        return new PolynomialKernel();
    case RBF:
        return new RbfKernel();
    default:
        break;
    }

    assert(false);
    return nullptr;
}
