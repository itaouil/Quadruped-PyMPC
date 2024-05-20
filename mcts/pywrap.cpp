#include "mcts.hpp"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(mcts_module, m) {
    py::class_<MCTS>(m, "MCTS")
        .def(py::init<float, int, int, int, bool, bool, bool, int>())
        .def("run", &MCTS::run)
        // .def("solve_batch_ocp", &MCTS::solveOCPs, py::call_guard<py::gil_scoped_release>())
        .def("set_current_state", &MCTS::setCurrentState);
}