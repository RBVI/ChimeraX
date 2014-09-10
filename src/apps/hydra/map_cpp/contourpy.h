#ifndef PYCONTOUR_HEADER_INCLUDED
#define  PYCONTOUR_HEADER_INCLUDED

extern "C" {

PyObject *surface_py(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *scale_and_shift_vertices(PyObject *, PyObject *args);
PyObject *scale_vertices(PyObject *, PyObject *args);
PyObject *shift_vertices(PyObject *, PyObject *args);
PyObject *affine_transform_vertices(PyObject *, PyObject *args);
PyObject *reverse_triangle_vertex_order(PyObject *, PyObject *args);

}

#endif
