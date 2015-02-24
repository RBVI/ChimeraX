// vi: set expandtab shiftwidth=4 softtabstop=4:
#ifndef OCCUPANCY_HEADER_INCLUDED
#define OCCUPANCY_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C" {

// Each xyz point contributes to the nearest 8 grid bins.
// Occupancy grid indices are ordered z, y, x and must be a NumPy float32 array.
PyObject *fill_occupancy_map(PyObject *s, PyObject *args, PyObject *keywds);

}

#endif
