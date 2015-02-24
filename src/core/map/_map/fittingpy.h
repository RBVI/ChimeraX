// vi: set expandtab shiftwidth=4 softtabstop=4:
#ifndef FITTINGPY_HEADER_INCLUDED
#define FITTINGPY_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C" {

PyObject *py_correlation_gradient(PyObject *, PyObject *args, PyObject *keywds);
PyObject *py_torque(PyObject *, PyObject *args, PyObject *keywds);
PyObject *py_torques(PyObject *, PyObject *args, PyObject *keywds);
PyObject *py_correlation_torque(PyObject *, PyObject *args, PyObject *keywds);
PyObject *py_correlation_torque2(PyObject *, PyObject *args, PyObject *keywds);

}

#endif
