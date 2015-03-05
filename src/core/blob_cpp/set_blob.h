// vi: set expandtab ts=4 sw=4:

#include <basegeom/Rgba.h>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>  // use PyArray_*(), NPY_*

namespace blob {

// set int
template<typename BlobType>
int
set_blob(PyObject* py_blob, PyObject* py_val,
    void (BlobType::MolType::* member_func)(int))
{
    if (py_val == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete C++ attribute");
        return -1;
    }

    if (PyArray_API == NULL)
        import_array1(-1);  // initialize NumPy
    BlobType* blob = static_cast<BlobType*>(py_blob);
    if (PyLong_Check(py_val)) {
        int val = (int)PyLong_AsLong(py_val);
        for (auto item: *(blob->_items))
            (item.get()->*member_func)(val);
        
    } else if (PyArray_Check(py_val)
    && PyArray_ISINTEGER((PyArrayObject*)py_val)) {
        PyArrayObject* array = PyArray_GETCONTIGUOUS((PyArrayObject*)py_val);
        if (PyArray_NDIM(array) != 1) {
            PyErr_SetString(PyExc_ValueError,
                "Numpy array of values must be one-dimensional");
            return -1;
        }
        if ((std::size_t)PyArray_DIMS(array)[0] != blob->_items->size()) {
            PyErr_SetString(PyExc_ValueError,
                "Size of numpy array does not match number of items to assign");
            return -1;
        }
        int item_size = PyArray_ITEMSIZE(array);
        if (item_size == sizeof(unsigned char)) {
            unsigned char* data = (unsigned char*) PyArray_DATA(array);
            for (auto item: *(blob->_items))
                (item.get()->*member_func)(*data++);
        } else if (item_size == sizeof(int)) {
            int* data = (int*) PyArray_DATA(array);
            for (auto item: *(blob->_items))
                (item.get()->*member_func)(*data++);
        } else if (item_size == sizeof(long)) {
            long* data = (long*) PyArray_DATA(array);
            for (auto item: *(blob->_items))
                (item.get()->*member_func)(*data++);
        } else {
            PyErr_SetString(PyExc_ValueError,
                "Array values must be byte, integer, or long");
            return -1;
        }
    } else {
        PyErr_SetString(PyExc_TypeError,
            "Value must be int or numpy array of int");
        return -1;
    }
    return 0;
}

// set bool
template<typename BlobType>
int
set_blob(PyObject* py_blob, PyObject* py_val,
    void (BlobType::MolType::* member_func)(bool))
{
    if (py_val == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete C++ attribute");
        return -1;
    }

    if (PyArray_API == NULL)
        import_array1(-1);  // initialize NumPy
    BlobType* blob = static_cast<BlobType*>(py_blob);
    if (PyBool_Check(py_val)) {
        bool val = py_val == Py_True;
        for (auto item: *(blob->_items))
            (item.get()->*member_func)(val);
        
    } else if (PyArray_Check(py_val)
    && PyArray_ISBOOL((PyArrayObject*)py_val)) {
        PyArrayObject* array = PyArray_GETCONTIGUOUS((PyArrayObject*)py_val);
        if (PyArray_NDIM(array) != 1) {
            PyErr_SetString(PyExc_ValueError,
                "Numpy array of values must be one-dimensional");
            return -1;
        }
        if ((std::size_t)PyArray_DIMS(array)[0] != blob->_items->size()) {
            PyErr_SetString(PyExc_ValueError,
                "Size of numpy array does not match number of items to assign");
            return -1;
        }
        int item_size = PyArray_ITEMSIZE(array);
        if (item_size == sizeof(signed char)) {
            signed char* data = (signed char*) PyArray_DATA(array);
            for (auto item: *(blob->_items))
                (item.get()->*member_func)(*data++);
        } else if (item_size == sizeof(int)) {
            int* data = (int*) PyArray_DATA(array);
            for (auto item: *(blob->_items))
                (item.get()->*member_func)(*data++);
        } else if (item_size == sizeof(long)) {
            long* data = (long*) PyArray_DATA(array);
            for (auto item: *(blob->_items))
                (item.get()->*member_func)(*data++);
        } else {
            PyErr_SetString(PyExc_ValueError,
                "Array values must be byte, integer, or long");
            return -1;
        }
    } else {
        PyErr_SetString(PyExc_TypeError,
            "Value must be bool or numpy array of bool");
        return -1;
    }
    return 0;
}

using basegeom::Rgba;

// set color (rgba)
template<typename BlobType>
int
set_blob(PyObject* py_blob, PyObject* py_val,
    void (BlobType::MolType::* member_func)(const Rgba&))
{
    if (py_val == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete C++ attribute");
        return -1;
    }

    if (PyArray_API == NULL)
        import_array1(-1);  // initialize NumPy
    BlobType* blob = static_cast<BlobType*>(py_blob);
    Rgba rgba;
    if (PyArray_Check(py_val) && PyArray_NDIM((PyArrayObject*)py_val) == 2) {
        PyArrayObject* array = PyArray_GETCONTIGUOUS((PyArrayObject*)py_val);
        if (!PyArray_ISINTEGER(array)
        || (std::size_t)PyArray_DIMS(array)[0] != blob->_items->size()
        || PyArray_DIMS(array)[1] != 4) {
            PyErr_SetString(PyExc_TypeError,
                "Value must be sequence of 4 ints or numpy Nx4 array of ints");
            return -1;
        }
        int item_size = PyArray_ITEMSIZE(array);
        if (item_size == sizeof(unsigned char)) {
            unsigned char* data = (unsigned char*) PyArray_DATA(array);
            for (auto item: *(blob->_items)) {
                rgba.r = *data++;
                rgba.g = *data++;
                rgba.b = *data++;
                rgba.a = *data++;
                (item.get()->*member_func)(rgba);
            }
        } else if (item_size == sizeof(unsigned int)) {
            unsigned int* data = (unsigned int*) PyArray_DATA(array);
            for (auto item: *(blob->_items)) {
                rgba.r = *data++;
                rgba.g = *data++;
                rgba.b = *data++;
                rgba.a = *data++;
                (item.get()->*member_func)(rgba);
            }
        } else if (item_size == sizeof(unsigned long)) {
            unsigned long* data = (unsigned long*) PyArray_DATA(array);
            for (auto item: *(blob->_items)) {
                rgba.r = *data++;
                rgba.g = *data++;
                rgba.b = *data++;
                rgba.a = *data++;
                (item.get()->*member_func)(rgba);
            }
        } else {
            PyErr_SetString(PyExc_ValueError,
                "Array values must be byte, integer, or long");
            return -1;
        }
    } else if (PySequence_Check(py_val) && PySequence_Size(py_val) == 4) {
        int vals[4];
        for (int i = 0; i < 4; ++i) {
            PyObject* item = PySequence_GetItem(py_val, i);
            if (!PyNumber_Check(item)) {
                PyErr_SetString(PyExc_TypeError,
                    "RGBA sequence items must be numbers");
                return -1;
            }
            PyObject* obj = PyNumber_Long(item);
            vals[i] = PyLong_AsLong(obj);
            Py_DECREF(obj);
        }
        rgba.r = vals[0];
        rgba.g = vals[1];
        rgba.b = vals[2];
        rgba.a = vals[3];
        for (auto item: *(blob->_items))
            (item.get()->*member_func)(rgba);
    } else {
        PyErr_SetString(PyExc_TypeError,
            "Value must be sequence of 4 ints or numpy Nx4 array of ints");
        return -1;
    }
    return 0;
}

// set float
template<typename BlobType>
int
set_blob(PyObject* py_blob, PyObject* py_val,
    void (BlobType::MolType::* member_func)(float))
{
    if (py_val == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete C++ attribute");
        return -1;
    }

    if (PyArray_API == NULL)
        import_array1(-1);  // initialize NumPy
    BlobType* blob = static_cast<BlobType*>(py_blob);
    if (PyFloat_Check(py_val)) {
        float val = (float)PyFloat_AS_DOUBLE(py_val);
        for (auto item: *(blob->_items))
            (item.get()->*member_func)(val);
        
    } else if (PyArray_Check(py_val)
    && PyArray_ISFLOAT((PyArrayObject*)py_val)) {
        PyArrayObject* array = PyArray_GETCONTIGUOUS((PyArrayObject*)py_val);
        if (PyArray_NDIM(array) != 1) {
            PyErr_SetString(PyExc_ValueError,
                "Numpy array of values must be one-dimensional");
            return -1;
        }
        if ((std::size_t)PyArray_DIMS(array)[0] != blob->_items->size()) {
            PyErr_SetString(PyExc_ValueError,
                "Size of numpy array does not match number of items to assign");
            return -1;
        }
        int item_size = PyArray_ITEMSIZE(array);
        if (item_size == sizeof(float)) {
            float* data = (float*) PyArray_DATA(array);
            for (auto item: *(blob->_items))
                (item.get()->*member_func)(*data++);
        } else if (item_size == sizeof(double)) {
            double* data = (double*) PyArray_DATA(array);
            for (auto item: *(blob->_items))
                (item.get()->*member_func)(*data++);
        } else {
            PyErr_SetString(PyExc_ValueError,
                "Array values must be float or double");
            return -1;
        }
    } else {
        PyErr_SetString(PyExc_TypeError,
            "Value must be float or numpy array of float");
        return -1;
    }
    return 0;
}

}  // namespace blob
