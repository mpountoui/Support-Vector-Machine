#include <vector>
#include "Vectors.hpp"
#include "Support_vector_machine.hpp"

/* ----------------------------------------------------------------------------------------- */

void LinearKernelTest()
{
    std::vector<Vector> X{ {1., 5.}, {2., 3.}, {3., 8.}, {4., 6.}, {5., 10.}, {6., 1.} };
    std::vector<int>    Y{ -1, -1, 1, 1, 1, -1 };

    SVM svm(LINEAR, 10, 0, 0, 0);

    svm.Fit(X, Y);
    auto const& a = svm.LagrangeMultipliers();

    printf("LinearKernelTest\n");
    std::fabs(a[0] - 0.20761245674721482) > 1e-16 ? printf("Error!\n") : printf("Ok!\n");
    std::fabs(a[1] - 0.00000000000000000) > 1e-16 ? printf("Error!\n") : printf("Ok!\n");
    std::fabs(a[2] - 0.00000000000000000) > 1e-16 ? printf("Error!\n") : printf("Ok!\n");
    std::fabs(a[3] - 0.28373702422125668) > 1e-16 ? printf("Error!\n") : printf("Ok!\n");
    std::fabs(a[4] - 0.00000000000000000) > 1e-16 ? printf("Error!\n") : printf("Ok!\n");
    std::fabs(a[5] - 0.07612456747404193) > 1e-16 ? printf("Error!\n") : printf("Ok!\n");
    printf("\n");
}

/* ----------------------------------------------------------------------------------------- */

void RbfKernelTest()
{
    std::vector<Vector> X{ {1., 5.}, {2., 5.}, {3., 8.}, {4., 3.}, {5., 10.}, {6., 1.} };
    std::vector<int>    Y{ -1, -1, 1, 1, 1, -1 };

    SVM svm(RBF, 10, 0, 2, 0);

    svm.Fit(X, Y);
    auto const& a = svm.LagrangeMultipliers();

    printf("RbfKernelTest\n");
    std::fabs(a[0] - 0.91724305168740428) > 1e-16 ? printf("Error!\n") : printf("Ok!\n");
    std::fabs(a[1] - 0.91724317872609507) > 1e-16 ? printf("Error!\n") : printf("Ok!\n");
    std::fabs(a[2] - 0.95862147695087085) > 1e-16 ? printf("Error!\n") : printf("Ok!\n");
    std::fabs(a[3] - 0.95862180335305147) > 1e-16 ? printf("Error!\n") : printf("Ok!\n");
    std::fabs(a[4] - 0.95862147505400863) > 1e-16 ? printf("Error!\n") : printf("Ok!\n");
    std::fabs(a[5] - 1.04137852494443162) > 1e-16 ? printf("Error!\n") : printf("Ok!\n");
    printf("\n");
}

/* ----------------------------------------------------------------------------------------- */

void PolyKernelTest()
{
    std::vector<Vector> X{ {1., 5.}, {2., 5.}, {3., 8.}, {4., 3.}, {5., 10.}, {6., 1.} };
    std::vector<int>    Y{ -1, -1, 1, 1, 1, -1 };

    SVM svm(POLYNOMIAL, 10, 3, 2, 1);

    svm.Fit(X, Y);
    auto const& a = svm.LagrangeMultipliers();

    printf("PolyKernelTest\n");
    std::fabs(a[0] - 0.00000000000000000) > 1e-16 ? printf("Error!\n") : printf("Ok!\n");
    std::fabs(a[1] - 0.00006041959571638) > 1e-16 ? printf("Error!\n") : printf("Ok!\n");
    std::fabs(a[2] - 0.00000789036934425) > 1e-16 ? printf("Error!\n") : printf("Ok!\n");
    std::fabs(a[3] - 0.00007759576132433) > 1e-16 ? printf("Error!\n") : printf("Ok!\n");
    std::fabs(a[4] - 0.00000000000000000) > 1e-16 ? printf("Error!\n") : printf("Ok!\n");
    std::fabs(a[5] - 0.00002506653495222) > 1e-16 ? printf("Error!\n") : printf("Ok!\n");
    printf("\n");
}

/* ----------------------------------------------------------------------------------------- */

int main()
{
    LinearKernelTest();
    RbfKernelTest();
    PolyKernelTest();
}