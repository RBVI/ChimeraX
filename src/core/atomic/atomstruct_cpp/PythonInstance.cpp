// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2016 Regents of the University of California.
 * All rights reserved.  This software provided pursuant to a
 * license agreement containing restrictions on its disclosure,
 * duplication and use.  For details see:
 * http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
 * This notice must be embedded in or attached to all copies,
 * including partial copies, of the software or any revisions
 * or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#include <sstream>
#include <stdexcept>

#include "Python.h"

#include "PythonInstance.h"

namespace {

class AcquireGIL {
    // RAII for Python GIL
    PyGILState_STATE gil_state;
public:
    AcquireGIL() {
        gil_state = PyGILState_Ensure();
    }
    ~AcquireGIL() {
        PyGILState_Release(gil_state);
    }
};

}

namespace atomstruct {

static PyObject* object_map_func = nullptr;

PyObject*
PythonInstance::get_py_attr(const char* attr_name) const
{
    auto py_obj = py_instance(this);
    if (py_obj == nullptr)
        throw NoPyInstanceError();

    auto py_attr = PyObject_GetAttrString(py_obj, attr_name);
    Py_DECREF(py_obj);
    if (py_attr == nullptr)
        throw NoPyAttrError();
    return py_attr;
}

double
PythonInstance::get_py_float_attr(const char* attr_name) const
{
    auto py_attr = get_py_attr(attr_name);
    if (!PyFloat_Check(py_attr)) {
        Py_DECREF(py_attr);
        std::stringstream msg;
        msg << "Expected Python attribute ";
        msg << attr_name;
        msg << " to be a float";
        throw WrongPyAttrTypeError(msg.str());
    }
    auto ret_val = PyFloat_AS_DOUBLE(py_attr);
    Py_DECREF(py_attr);
    return ret_val;
}

long
PythonInstance::get_py_int_attr(const char* attr_name) const
{
    auto py_attr = get_py_attr(attr_name);
    if (!PyLong_Check(py_attr)) {
        Py_DECREF(py_attr);
        std::stringstream msg;
        msg << "Expected Python attribute ";
        msg << attr_name;
        msg << " to be an int";
        throw WrongPyAttrTypeError(msg.str());
    }
    auto ret_val = PyLong_AsLong(py_attr);
    Py_DECREF(py_attr);
    return ret_val;
}

static std::string _buffer;  // so that the const char* will hang around
const char *
PythonInstance::get_py_string_attr(const char* attr_name) const
{
    auto py_attr = get_py_attr(attr_name);
    if (!PyUnicode_Check(py_attr)) {
        Py_DECREF(py_attr);
        std::stringstream msg;
        msg << "Expected Python attribute ";
        msg << attr_name;
        msg << " to be a string";
        throw WrongPyAttrTypeError(msg.str());
    }
    _buffer = PyUnicode_AsUTF8(py_attr);
    Py_DECREF(py_attr);
    return _buffer.c_str();
}

PyObject*
PythonInstance::py_instance(const void* ptr)
{
    AcquireGIL gil;  // guarantee that we can call Python functions
    if (object_map_func == nullptr) {
        auto molobject_module = PyImport_ImportModule("chimerax.core.atomic.molobject");
        if (molobject_module == nullptr)
            throw std::runtime_error("Cannot import chimerax.core.atomic.molobject module");
        object_map_func = PyObject_GetAttrString(molobject_module, "object_map");
        if (object_map_func == nullptr) {
            Py_DECREF(molobject_module);
            throw std::runtime_error("Cannot get object_map() func from"
                " chimerax.atomic.molobject module");
        }
        Py_DECREF(molobject_module);
        if (!PyCallable_Check(object_map_func)) {
            Py_DECREF(object_map_func);
            object_map_func = nullptr;
            throw std::runtime_error("chimerax.core.atomic.molobject.object_map is not callable");
        }
    }
    auto arg_tuple = PyTuple_New(2);
    if (arg_tuple == nullptr) {
        throw std::runtime_error("Could not create arg tuple to call"
            " chimerax.core.atomic.molobject.object_map with");
    }
    auto py_ptr = PyLong_FromVoidPtr(const_cast<void*>(ptr));
    if (py_ptr == nullptr) {
        Py_DECREF(arg_tuple);
        throw std::runtime_error("Could not convert pointer to Python long"
            " to use as arg to chimerax.core.atomic.molobject.object_map");
    }
    PyTuple_SET_ITEM(arg_tuple, 0, py_ptr);
    PyTuple_SET_ITEM(arg_tuple, 1, Py_None);
    auto ret_val = PyObject_CallObject(object_map_func, arg_tuple);
    Py_DECREF(arg_tuple);
    if (ret_val == nullptr) {
        throw std::runtime_error("Calling chimerax.core.atomic.molobject.object_map failed");
    }
    if (ret_val == Py_None) {
        Py_DECREF(Py_None);
        return nullptr;
    }
    return ret_val;
}

} //  namespace atomstruct
