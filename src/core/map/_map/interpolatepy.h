// vi: set expandtab shiftwidth=4 softtabstop=4:
#ifndef PYINTERPOLATE_HEADER_INCLUDED
#define PYINTERPOLATE_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C" {

PyObject *interpolate_volume_data(PyObject *, PyObject *args);
PyObject *interpolate_volume_gradient(PyObject *, PyObject *args);
PyObject *interpolate_colormap(PyObject *, PyObject *args);
PyObject *set_outside_volume_colors(PyObject *, PyObject *args);

}

#endif
