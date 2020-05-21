
#include <pybind11/pybind11.h>
#include "./_sample.h"

namespace py=pybind11;

PYBIND11_MODULE(_sample_pybind11, m) {
    m.doc() = _sample_module_doc;
    // PyBind11 allows you to compose methods on-the-fly using C++11 lambda functions
    m.def("counts", [](size_t struct_pointer)
    {
        return atom_and_bond_count((void *)struct_pointer);
    },
    _sample_counts_doc);
}
