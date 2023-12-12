#include <pybind11/pybind11.h>
#include "Support_vector_machine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mySVM, m) {
    py::class_<SVM>(m, "SVM")
    .def(py::init<>())
    .def("Test", &SVM::Test);
}
