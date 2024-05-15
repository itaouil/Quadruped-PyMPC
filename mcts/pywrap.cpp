#include "mcts.hpp"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(mcts_module, m) {
    py::class_<MCTS>(m, "MCTS")
        .def(py::init<float, int, int, int, bool, bool, bool>())
        .def("run", &MCTS::run)
        .def("set_current_state", &MCTS::setCurrentState);
}