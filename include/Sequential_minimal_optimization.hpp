#ifndef SEQUENTIAL_MINIMAL_OTIMIZATION
#define SEQUENTIAL_MINIMAL_OTIMIZATION

#include <vector>
#include <unordered_map>
#include "Support_vector_machine.hpp"

/* ----------------------------------------------------------------------------------------- */

extern double epsilon;

/* ----------------------------------------------------------------------------------------- */

class DualProblem
{
private:
    SVM& m_SVM;
    std::unordered_map<size_t, double> m_Errors;

private:
    void UpdateLagrangeMultipliers(size_t, size_t, double, double) const;

    bool UpdateErros();
    bool PerformStep(size_t, size_t, double);
    bool ExamineExample(size_t);

    size_t ChooseSecondLagrangeMultiplier(double) const;
    double GetError(size_t) const;

    std::pair<double, double> EndsOfLine(double, double, bool) const;

public:
    DualProblem(SVM&);
    DualProblem(DualProblem const&)            = delete;
    DualProblem(DualProblem&&)                 = delete;
    DualProblem& operator=(DualProblem const&) = delete;
    DualProblem& operator=(DualProblem&&)      = delete;
    ~DualProblem()                             = default;

public:
    bool Solve();
};

#endif