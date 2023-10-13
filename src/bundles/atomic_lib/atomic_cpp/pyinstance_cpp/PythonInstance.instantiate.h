// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * This software is provided pursuant to the ChimeraX license agreement, which
 * covers academic and commercial uses. For more information, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This file is part of the ChimeraX library. You can also redistribute and/or
 * modify it under the GNU Lesser General Public License version 2.1 as
 * published by the Free Software Foundation. For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * This file is distributed WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
 * must be embedded in or attached to all copies, including partial copies, of
 * the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#ifndef pyinstance_python_instance_instantiate
#define pyinstance_python_instance_instantiate

#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>

#include "Python.h"

namespace pyinstance {

template <class C> std::string PythonInstance<C>::_buffer;
template <class C> PyObject* PythonInstance<C>::_py_class = nullptr;

template <class C>
double PythonInstance<C>::get_py_float_attr(std::string& attr_name, bool create) const
{
    return get_py_float_attr(attr_name.c_str(), create);
}

template <class C>
long PythonInstance<C>::get_py_int_attr(std::string& attr_name, bool create) const
{
    return get_py_int_attr(attr_name.c_str(), create);
}

template <class C>
const char* PythonInstance<C>::get_py_string_attr(std::string& attr_name, bool create) const
{
    return get_py_string_attr(attr_name.c_str(), create);
}
    
template <class C>
PyObject* PythonInstance<C>::py_class()
{
    return _py_class;
}

template <class C>
void PythonInstance<C>::set_py_class(PyObject* c_obj)
{
    Py_INCREF(c_obj);
    _py_class = c_obj;
}

template <class C>
void PythonInstance<C>::set_py_instance(PyObject* py_obj)
{
    auto derived = static_cast<const C*>(this);
    _pyinstance_object_map[derived] = py_obj;
    Py_INCREF(py_obj);
}

template <class C>
PythonInstance<C>::~PythonInstance() {
    if (!Py_IsInitialized())
        return;
    auto derived = static_cast<const C*>(this);
    auto i = _pyinstance_object_map.find(static_cast<const void*>(derived));
    if (i == _pyinstance_object_map.end())
        return;
    PyObject* py_inst = (*i).second;
    AcquireGIL gil; // Py_DECREF can cause code to run
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
    if (py_obj == Py_None) {
        Py_DECREF(py_obj);
        throw NoPyInstanceError();
    }

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
    // Returns a new reference
    auto derived = static_cast<const C*>(this);
    auto i = _pyinstance_object_map.find(static_cast<const void*>(derived));
    if (i != _pyinstance_object_map.end()) {
        Py_INCREF(i->second);
        return i->second;
    }

    if (!create) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    PyObject* class_inst = py_class();
    if (class_inst == nullptr) {
        std::stringstream msg;
        msg << "Cannot instantiate Python class corresponding to C++ ";
        msg << typeid(*derived).name();
        throw std::invalid_argument(msg.str());
    }
    
    AcquireGIL gil;  // guarantee that we can call Python functions
    PyObject* py_ptr = PyLong_FromVoidPtr(const_cast<void*>(static_cast<const void*>(derived)));
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
    _pyinstance_object_map[derived] = py_inst;
    Py_INCREF(py_inst);
    return py_inst;
}

template <class C>
PyObject*
PythonInstance<C>::py_call_method(const std::string& method_name, const char* fmt, const void* arg) const
// limited to 0 or 1 arg methods
{
    auto inst = py_instance(false);
    PyObject* ret;
    if (inst == Py_None) {
        return nullptr;
    } else {
        auto gil = AcquireGIL();
        ret = PyObject_CallMethod(inst, method_name.c_str(), fmt, arg);
        if (ret == nullptr) {
            std::stringstream msg;
            msg << "Calling " << py_class_name() << " " << method_name << " failed.";
            throw std::runtime_error(msg.str());
        }
    }
    Py_DECREF(inst);
    return ret;
}

template <class C>
std::string
PythonInstance<C>::py_class_name() const
{
    std::stringstream msg;
    PyObject* class_inst = py_class();
    if (class_inst == nullptr) {
        auto derived = static_cast<const C*>(this);
        msg << "[C++: " << typeid(*derived).name() << "]";
        return msg.str();
    }

    PyObject* class_name = PyObject_GetAttrString(class_inst, "__name__");
    if (class_name == nullptr)
        throw std::runtime_error("Cannot get class __name__ attr in C++");
    msg << PyUnicode_AsUTF8(class_name);
    Py_DECREF(class_name);
    return msg.str();
}

}  // namespace pyinstance

#endif  // pyinstance_python_instance_instantiate
