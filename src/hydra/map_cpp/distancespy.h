#ifndef PYDISTANCE_HEADER_INCLUDED
#define PYDISTANCE_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C" {

PyObject *py_distances_from_origin(PyObject *, PyObject *args);
PyObject *py_distances_perpendicular_to_axis(PyObject *, PyObject *args);
PyObject *py_distances_parallel_to_axis(PyObject *, PyObject *args);
PyObject *py_maximum_norm(PyObject *, PyObject *args, PyObject *keywds);
PyObject *py_correlation_gradient(PyObject *, PyObject *args, PyObject *keywds);
PyObject *py_torque(PyObject *, PyObject *args, PyObject *keywds);
PyObject *py_torques(PyObject *, PyObject *args, PyObject *keywds);
PyObject *py_correlation_torque(PyObject *, PyObject *args, PyObject *keywds);
PyObject *py_correlation_torque2(PyObject *, PyObject *args, PyObject *keywds);

}

#endif
