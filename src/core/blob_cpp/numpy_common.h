// vi: set expandtab ts=4 sw=4:
#ifndef blob_numpy_common
#define blob_numpy_common

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>  // use PyArray_*(), NPY_*

namespace blob {
    
void* initialize_numpy();
extern PyObject* allocate_python_array(
        unsigned int dim, unsigned int *size, int type);
extern PyObject* allocate_python_array(
        unsigned int dim, unsigned int *size, PyArray_Descr *dtype);

}  // namespace blob

#endif  // blob_numpy_common
