#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "Support_vector_machine.hpp"

/* ----------------------------------------------------------------------------------------- */

namespace py = pybind11;

/* ----------------------------------------------------------------------------------------- */

class pythonSVM : public SVM
{
private:
    KernelType StringToType(std::string const &) const;

public:
    pythonSVM(std::string const &, double, size_t, double, double);
    pythonSVM(pythonSVM const&)            = delete;
    pythonSVM(pythonSVM&&)                 = delete;
    pythonSVM& operator=(pythonSVM const&) = delete;
    pythonSVM& operator=(pythonSVM&&)      = delete;
    ~pythonSVM()                           = default;

public:
    py::array_t<int> Predict(std::vector<Vector> const &x_vec) const;
};

/* ----------------------------------------------------------------------------------------- */

PYBIND11_MODULE(mySVM, m)
{
    py::class_<pythonSVM>(m, "SVM")
        .def(py::init<std::string const&, double, size_t, double, double>(), 
        py::arg("Kernel"), py::arg("C") = 10.0, py::arg("degree") = 2, py::arg("gamma") = 2.0, py::arg("coef0") = 2.0)
        .def("fit", &pythonSVM::Fit)
        .def("predict", &pythonSVM::Predict);
}

/* ----------------------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------------------- */

pythonSVM::pythonSVM(std::string const & str, double C, size_t degree, double gamma, double Coeff)
:
SVM( StringToType(str), C, degree, gamma, Coeff )
{}

/* ----------------------------------------------------------------------------------------- */

KernelType pythonSVM::StringToType(std::string const &str) const
{
    if (str == "linear")
    {
        return LINEAR;
    }
    else if (str == "poly")
    {
        return POLYNOMIAL;
    }
    else if (str == "rbf")
    {
        return RBF;
    }

    assert(false);
    return LINEAR;
}

/* ----------------------------------------------------------------------------------------- */

py::array_t<int> pythonSVM::Predict(std::vector<Vector> const &x_vec) const
{
    auto y_vec = SVM::Predict(x_vec);
    return py::array_t<int>(y_vec.size(), y_vec.data());
}
