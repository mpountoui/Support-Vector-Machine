#include <cstdio>
#include <vector>
#include "SupportVectorMachine.hpp"

class QuadraticProgrammingProblem
{
private:
    SVM& m_SVM;

private:
    bool PerformStep(size_t, size_t) const;

public:
    QuadraticProgrammingProblem(SVM&);
    QuadraticProgrammingProblem(QuadraticProgrammingProblem const&)            = delete;
    QuadraticProgrammingProblem(QuadraticProgrammingProblem&&)                 = delete;
    QuadraticProgrammingProblem& operator=(QuadraticProgrammingProblem const&) = delete;
    QuadraticProgrammingProblem& operator=(QuadraticProgrammingProblem&&)      = delete;
    ~QuadraticProgrammingProblem()                                             = default;

public:
    bool Solve() const;
};
