// vi: set expandtab ts=4 sw=4:

#include <Python.h>
#include <vector>
#include <memory>
#include "imex.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>  // use PyArray_*(), NPY_*

namespace blob {
    
template<typename BlobType>
PyObject*
blob_filter(PyObject* self, PyObject* bools)
{
    if (PyArray_API == NULL)
        import_array1(NULL);  // initialize NumPy
    PyObject* bool_array = PyArray_FromAny(bools,
        PyArray_DescrFromType(NPY_BOOL), 1, 1, NPY_ARRAY_CARRAY_RO, NULL);
    if (bool_array == NULL) {
        PyErr_SetString(PyExc_TypeError,
            "Value cannot be converted to numpy array of bool");
        return NULL;
    }
    PyArrayObject* array = PyArray_GETCONTIGUOUS((PyArrayObject*)bool_array);
    BlobType* blob = static_cast<BlobType*>(self);
    if ((std::size_t)PyArray_DIMS(array)[0] != blob->_items->size()) {
        PyErr_SetString(PyExc_ValueError,
            "Size of numpy array does not match number of items to assign");
        return NULL;
    }
    int item_size = PyArray_ITEMSIZE(array);
    if (item_size == sizeof(signed char)) {
        BlobType* filtered = static_cast<BlobType*>(
                new_blob<BlobType>(self->ob_type));
        signed char* data = (signed char*) PyArray_DATA(array);
        for (auto item: *(blob->_items))
            if (*data++ == 1)
                filtered->_items->emplace_back(item.get());
        return filtered;
    } else {
        PyErr_SetString(PyExc_ValueError,
            "Array values must be byte, integer, or long");
        return NULL;
    }
}

}  // namespace blob
