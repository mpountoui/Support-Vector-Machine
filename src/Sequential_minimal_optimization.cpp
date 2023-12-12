#include <iostream>
#include <cstdio>
#include <cassert>
#include "Sequential_minimal_optimization.hpp"

/* ----------------------------------------------------------------------------------------- */

QuadraticProgrammingProblem::QuadraticProgrammingProblem(SVM& SVM_in)
    :
    m_SVM(SVM_in)
{}

/* ----------------------------------------------------------------------------------------- */

bool QuadraticProgrammingProblem::PerformStep(size_t i, size_t k) const
{
    if(i == k){ assert(false); return false; }
    double a_1 = m_SVM.m_LagrMult[i];
    double y_1 = m_SVM.m_Targets[i];

    return true;
}

/* ----------------------------------------------------------------------------------------- */

bool QuadraticProgrammingProblem::Solve() const
{
    return false;
}

/* ----------------------------------------------------------------------------------------- */

int main()
{
    std::cout << "Test file!" << std::endl;
}
