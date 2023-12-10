#include <vector>

class Kernel
{
public:
    Kernel()                         = default;
    Kernel(Kernel const&)            = delete;
    Kernel(Kernel&&)                 = delete;
    Kernel& operator=(Kernel const&) = delete;
    Kernel& operator=(Kernel&&)      = delete;
    ~Kernel()                        = default;

public:
    double operator()(std::vector<double> const&, std::vector<double> const&) const;
};
