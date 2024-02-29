#ifndef KERNEL
#define KERNEL

#include <vector>
#include "Vectors.hpp"

/* ----------------------------------------------------------------------------------------- */

enum KernelType
{
    LINEAR,
    POLYNOMIAL,
    RBF
};

/* ----------------------------------------------------------------------------------------- */

struct SupportVectors
{
    int    y = 0;
    double a = 0.0;
    Vector x;
};

/* ----------------------------------------------------------------------------------------- */

class Kernel
{
protected:
    Kernel()                         = default;
    Kernel(Kernel const&)            = default;
    Kernel(Kernel&&)                 = default;
    Kernel& operator=(Kernel const&) = default;
    Kernel& operator=(Kernel&&)      = default;

public: 
    virtual ~Kernel()                = default;

public:
    virtual void W_InitializeIfLinearKernel(std::vector<int>    const&,
                                            std::vector<double> const&,
                                            std::vector<Vector> const&) = 0;

    virtual void W_UpdateIfLinearKernel(int, double, Vector const&,
                                        int, double, Vector const&) = 0;

    virtual void SaveSupportVectorsIfNonLinear(std::vector<int>    const&,
                                               std::vector<double> const&,
                                               std::vector<Vector> const&) = 0;

    virtual double Predict(Vector const&, double) = 0;

    virtual double operator()(Vector const&, Vector const&) const = 0;

    virtual Kernel* clone() const = 0;
};

/* ----------------------------------------------------------------------------------------- */

Kernel* KernelFactory(KernelType, size_t, double, double);

#endif