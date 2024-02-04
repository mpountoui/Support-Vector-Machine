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
public:
    py::array_t<int> Predict(std::vector<Vector> const& x_vec) const
    {
        auto y_vec = SVM::Predict(x_vec);
        return py::array_t<int>( y_vec.size() , y_vec.data() );
    }
};

/* ----------------------------------------------------------------------------------------- */

PYBIND11_MODULE(mySVM, m) {
    py::class_<pythonSVM>(m, "SVM")
    .def(py::init<>())
    .def("fit"    , &pythonSVM::Fit    )
    .def("predict", &pythonSVM::Predict);
}