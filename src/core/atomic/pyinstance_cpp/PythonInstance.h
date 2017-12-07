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

#ifndef pyinstance_python_instance
#define pyinstance_python_instance

#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>

#include "Python.h"

#include "imex.h"

namespace pyinstance {

class PYINSTANCE_IMEX AcquireGIL {
public:
    AcquireGIL();
    ~AcquireGIL();
};

class PyAttrError : public std::runtime_error {
public:
    PyAttrError(const std::string msg) : std::runtime_error(msg) {}
};

class NoPyInstanceError : public PyAttrError {
public:
    NoPyInstanceError() : PyAttrError(std::string("No Python instance")) {}
    NoPyInstanceError(const std::string msg) : PyAttrError(msg) {}
};

class NoPyAttrError : public PyAttrError {
public:
    NoPyAttrError() : PyAttrError(std::string("Python instance has no such attr")) {}
    NoPyAttrError(const std::string msg) : PyAttrError(msg) {}
};

class WrongPyAttrTypeError : public PyAttrError {
public:
    WrongPyAttrTypeError() : PyAttrError(std::string("Python attr is wrong type")) {}
    WrongPyAttrTypeError(const std::string msg) : PyAttrError(msg) {}
};

PYINSTANCE_IMEX extern std::map<const void*, PyObject*>  _pyinstance_object_map;

// this is a template class so that different derived classes have separate static variables
template<class C>
class PYINSTANCE_IMEX PythonInstance {
private:
    static std::string _buffer;  // so that the const char* from std::string will hang around
    static PyObject*  _py_class;
public:
    virtual  ~PythonInstance();
    PyObject*  get_py_attr(const char* attr_name, bool create=false) const;
    double  get_py_float_attr(const char* attr_name, bool create=false) const;
    long  get_py_float_attr(std::string& attr_name, bool create=false) const {
        return get_py_float_attr(attr_name.c_str(), create);
    }
    long  get_py_int_attr(const char* attr_name, bool create=false) const;
    long  get_py_int_attr(std::string& attr_name, bool create=false) const {
        return get_py_int_attr(attr_name.c_str(), create);
    }
    const char*  get_py_string_attr(const char* attr_name, bool create=false) const;
    const char*  get_py_string_attr(std::string& attr_name, bool create=false) const {
        return get_py_string_attr(attr_name.c_str(), create);
    }
    
    static PyObject*  py_class() { return _py_class; }
    static void  set_py_class(PyObject* c_obj) { _py_class = c_obj; }

    PyObject*  py_instance(bool create) const;
    // some Python objects can't be created by C++ (need more args), so...
    void  set_py_instance(PyObject* py_obj) {
        _pyinstance_object_map[this] = py_obj;
        Py_INCREF(py_obj);
    }
};

template <class C>
PyObject*  PythonInstance<C>::_py_class = nullptr;

template <class C>
std::string  PythonInstance<C>::_buffer;

template <class C>
PythonInstance<C>::~PythonInstance() {
    auto i = _pyinstance_object_map.find(static_cast<const void*>(this));
    if (i != _pyinstance_object_map.end())
        return;
    PyObject* py_inst = (*i).second;
    PyObject_DelAttrString(py_inst, "_c_pointer");
    PyObject_DelAttrString(py_inst, "_c_pointer_ref");
    Py_DECREF(py_inst);
    _pyinstance_object_map.erase(i);
}

template <class C>
PyObject*
PythonInstance<C>::get_py_attr(const char* attr_name, bool create) const
{
    auto py_obj = py_instance(create);
    if (py_obj == nullptr)
        throw NoPyInstanceError();

    auto py_attr = PyObject_GetAttrString(py_obj, attr_name);
    Py_DECREF(py_obj);
    if (py_attr == nullptr) {
        PyErr_Clear();
        throw NoPyAttrError();
    }
    return py_attr;
}

template <class C>
double
PythonInstance<C>::get_py_float_attr(const char* attr_name, bool create) const
{
    auto py_attr = get_py_attr(attr_name, create);
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

template <class C>
long
PythonInstance<C>::get_py_int_attr(const char* attr_name, bool create) const
{
    auto py_attr = get_py_attr(attr_name, create);
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

template <class C>
const char *
PythonInstance<C>::get_py_string_attr(const char* attr_name, bool create) const
{
    auto py_attr = get_py_attr(attr_name, create);
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

template <class C>
PyObject*
PythonInstance<C>::py_instance(bool create) const
{
    auto i = _pyinstance_object_map.find(static_cast<const void*>(this));
    if (i != _pyinstance_object_map.end())
        return (*i).second;

    if (!create) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    PyObject* class_inst = py_class();
    if (class_inst == nullptr) {
        std::stringstream msg;
        msg << "Cannot instantiate Python class corresponding to C++ ";
        msg << typeid(*this).name();
        throw std::invalid_argument(msg.str());
    }
    
    AcquireGIL gil;  // guarantee that we can call Python functions
    PyObject* py_ptr = PyLong_FromVoidPtr(const_cast<void*>(static_cast<const void*>(this)));
    PyObject* py_inst = PyObject_CallFunctionObjArgs(class_inst, py_ptr, nullptr);
    Py_DECREF(py_ptr);
    if (py_inst == nullptr) {
        PyObject* class_name = PyObject_GetAttrString(class_inst, "__name__");
        if (class_name == nullptr)
            throw std::runtime_error("Cannot get class __name__ attr in C++");
        std::stringstream msg;
        msg << "Cannot construct Python " << PyUnicode_AsUTF8(class_name) << " instance from C++ ";
        Py_DECREF(class_name);
        throw std::runtime_error(msg.str());
    }
    _pyinstance_object_map[this] = py_inst;
    Py_INCREF(py_inst);
    return py_inst;
}

}  // namespace pyinstance

#endif  // pyinstance_python_instance
