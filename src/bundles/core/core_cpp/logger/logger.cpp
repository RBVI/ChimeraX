// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This particular file is part of the ChimeraX library. You can also
 * redistribute and/or modify it under the terms of the GNU Lesser General
 * Public License version 2.1 as published by the Free Software Foundation.
 * For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
 * LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
 * VERSION 2.1
 *
 * This notice must be embedded in or attached to all copies, including partial
 * copies, of the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#include <cctype>
#include <iostream>
#include <stdexcept>
#include <string>

#include "Python.h"
#define LOGGER_EXPORT
#include "logger.h"

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

namespace logger {

static std::string
py_exception_msg()
{
    PyObject* err_type;
    PyObject* err_val;
    PyObject* err_trace;
    PyErr_Fetch(&err_type, &err_val, &err_trace);
    PyObject* val_str = PyObject_Str(err_val);
    PyErr_Restore(err_type, err_val, err_trace);
    auto utf8_val = PyUnicode_AsUTF8(val_str);
    Py_DECREF(val_str);
    return std::string(utf8_val);
}

void  _log(PyObject* logger, std::stringstream& msg, _LogLevel level, bool is_html)
{
    if (logger == nullptr || logger == Py_None) {
        if (level == _LogLevel::ERROR)
            std::cerr << msg.str() << "\n";
        else
            std::cout << msg.str() << "\n";
    } else {
        // Python logging
        AcquireGIL gil;   // guarantee that we can call Python functions

        const char* method_name;
        if (level == _LogLevel::ERROR)
            method_name = "error";
        else if (level == _LogLevel::WARNING)
            method_name = "warning";
        else if (level == _LogLevel::INFO)
            method_name = "info";
        else
            method_name = "status";
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
        std::string sanitized_string;
        for (auto c: msg.str()) {
            if (std::isprint(c) || std::isspace(c))
                sanitized_string.push_back(c);
            else
                sanitized_string.push_back('?');
        }
        PyObject* py_msg = PyUnicode_FromString(sanitized_string.c_str());
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
        PyObject* kw = PyDict_New();
        if (kw == nullptr) {
            Py_DECREF(args);
            std::stringstream err_msg;
            err_msg << "Could not create kw dict for calling logger method: "
                << py_exception_msg();
            throw std::runtime_error(err_msg.str());
        }
        if (PyDict_SetItemString(kw, "is_html", is_html ? Py_True : Py_False) < 0) {
            Py_DECREF(args);
            std::stringstream err_msg;
            err_msg << "Could not insert 'is_html' into kw dict for calling logger method: "
                << py_exception_msg();
            throw std::runtime_error(err_msg.str());
        }
        PyObject* retval = PyObject_Call(method, args, kw);
        if (retval == nullptr) {
            Py_DECREF(args);
            Py_DECREF(kw);
            std::stringstream err_msg;
            err_msg << "Call to logger '" << method_name << "' method failed: "
                << py_exception_msg();
            throw std::runtime_error(err_msg.str());
        }
        Py_DECREF(args);
        Py_DECREF(kw);
    }
}

} // namespace logger
