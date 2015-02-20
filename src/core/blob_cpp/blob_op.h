// vi: set expandtab ts=4 sw=4:

#include <Python.h>
#include <memory>
#include <unordered_set>
#include <vector>
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

template<typename BlobType>
PyObject*
blob_merge(PyObject* self, PyObject* py_other_blob)
{
    if (self->ob_type != py_other_blob->ob_type) {
        PyErr_SetString(PyExc_ValueError, "Merged blobs must be same type");
        return NULL;
    }
    BlobType* my_blob = static_cast<BlobType*>(self);
    BlobType* other_blob = static_cast<BlobType*>(py_other_blob);
    BlobType* merged = static_cast<BlobType*>(
        new_blob<BlobType>(self->ob_type));

    merged->_items->insert(merged->_items->end(),
        my_blob->_items->begin(), my_blob->_items->end());
    std::unordered_set<typename BlobType::MolType*>
        seen(my_blob->_items->size());
    for (auto i: *(my_blob->_items)) {
        seen.insert(i.get());
    }
    for (auto i: *(other_blob->_items)) {
        if (seen.find(i.get()) != seen.end())
            merged->_items->emplace_back(i);
    }
    return merged;
}

}  // namespace blob
