#include <vector>
#include "Vectors.hpp"
#include "Support_vector_machine.hpp"

/* ----------------------------------------------------------------------------------------- */

void Test()
{    
    std::vector<Vector> X{ {1., 5.}, {2., 3.}, {3., 8.}, {4., 6.}, {5., 10.}, {6., 1.} };
    std::vector<int>    Y{ -1, -1, 1, 1, 1, -1 };

    SVM svm;
    svm.Fit(X, Y);
    auto const& a = svm.LagrangeMultipliers();

    std::fabs(a[0] - 0.20758503929671218) > 1e-16 ? printf("Error!\n") : printf("Ok!\n");
    std::fabs(a[1] - 0.00000000000000000) > 1e-16 ? printf("Error!\n") : printf("Ok!\n");
    std::fabs(a[2] - 0.00000000000000000) > 1e-16 ? printf("Error!\n") : printf("Ok!\n");
    std::fabs(a[3] - 0.28370866134142636) > 1e-16 ? printf("Error!\n") : printf("Ok!\n");
    std::fabs(a[4] - 0.00000000000000000) > 1e-16 ? printf("Error!\n") : printf("Ok!\n");
    std::fabs(a[5] - 0.07612362204471422) > 1e-16 ? printf("Error!\n") : printf("Ok!\n");
}

/* ----------------------------------------------------------------------------------------- */

int main()
{
    Test();
}