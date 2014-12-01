#ifndef TRANSFORM_HEADER_INCLUDED
#define  TRANSFORM_HEADER_INCLUDED

extern "C" {

PyObject *scale_and_shift_vertices(PyObject *, PyObject *args);
PyObject *scale_vertices(PyObject *, PyObject *args);
PyObject *shift_vertices(PyObject *, PyObject *args);
PyObject *affine_transform_vertices(PyObject *, PyObject *args);

}

#endif
