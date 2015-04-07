// vi: set expandtab ts=4 sw=4:

#include <iostream>
#include <stdexcept>
#include <string>

#include "Python.h"
#include "logger.h"

namespace logger {

static std::string
py_exception_msg()
{
    PyObject* err_type;
    PyObject* err_val;
    PyObject* err_trace;
    PyErr_Fetch(&err_type, &err_val, &err_trace);
    PyErr_Restore(err_type, err_val, err_trace);
    return std::string(PyUnicode_AsUTF8(err_val));
}

// placeholders until the Python Logger is implemented...
void  _log(PyObject* logger, std::stringstream& msg, _LogLevel level)
{
    if (logger == nullptr) {
        if (level == _LogLevel::ERROR)
            std::cerr << msg.str() << "\n";
        else
            std::cout << msg.str() << "\n";
    } else {
        // Python logging
        const char* method_name;
        if (level == _LogLevel::ERROR)
            method_name = "error";
        else if (level == _LogLevel::WARNING)
            method_name = "warning";
        else
            method_name = "info";
        PyObject* method = PyObject_GetAttrString(logger, method_name);
        if (method == nullptr) {
            std::stringstream err_msg;
            err_msg << "No '" << method_name << "' method in logger object.";
            throw std::invalid_argument(err_msg.str());
        }
        if (!PyCallable_Check(method)) {
            std::stringstream err_msg;
            err_msg << "'" << method_name <<
                "' method in logger object is not callable.";
            throw std::invalid_argument(err_msg.str());
        }
        PyObject* py_msg = PyUnicode_FromString(msg.str().c_str());
        if (py_msg == nullptr) {
            std::stringstream err_msg;
            err_msg << "Could not convert error message to unicode: "
                << py_exception_msg();
            throw std::runtime_error(err_msg.str());
        }
        PyObject* args = PyTuple_Pack(1, py_msg);
        if (args == nullptr) {
            std::stringstream err_msg;
            err_msg << "Could not make arg tuple for calling logger method: "
                << py_exception_msg();
            throw std::runtime_error(err_msg.str());
        }
        PyObject* retval = PyObject_CallObject(method, args);
        if (retval == nullptr) {
            std::stringstream err_msg;
            err_msg << "Call to logger '" << method_name << "' method failed: "
                << py_exception_msg();
            throw std::runtime_error(err_msg.str());
        }
    }
}

} // namespace logger
