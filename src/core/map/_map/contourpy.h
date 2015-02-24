// vi: set expandtab shiftwidth=4 softtabstop=4:
#ifndef PYCONTOUR_HEADER_INCLUDED
#define  PYCONTOUR_HEADER_INCLUDED

extern "C" {

PyObject *surface_py(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *reverse_triangle_vertex_order(PyObject *, PyObject *args);

}

#endif
