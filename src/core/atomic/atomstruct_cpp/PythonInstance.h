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

#ifndef atomstruct_python_instance
#define atomstruct_python_instance

#include <stdexcept>
#include <string>

#include "imex.h"

// "forward declare" PyObject, which is a typedef of a struct,
// as per the python mailing list:
// http://mail.python.org/pipermail/python-dev/2003-August/037601.html
#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif

namespace atomstruct {

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

class ATOMSTRUCT_IMEX PythonInstance {
public:
    PyObject*  get_py_attr(const char* attr_name) const;
    double  get_py_float_attr(const char* attr_name) const;
    long  get_py_float_attr(std::string& attr_name) const {
        return get_py_float_attr(attr_name.c_str());
    }
    long  get_py_int_attr(const char* attr_name) const;
    long  get_py_int_attr(std::string& attr_name) const {
        return get_py_int_attr(attr_name.c_str());
    }
    const char*  get_py_string_attr(const char* attr_name) const;
    const char*  get_py_string_attr(std::string& attr_name) const {
        return get_py_string_attr(attr_name.c_str());
    }
    
    static PyObject*  py_instance(const void* ptr);
    PyObject*  py_instance() const { return py_instance(this); }
};



}  // namespace atomstruct

#endif  // atomstruct_python_instance
