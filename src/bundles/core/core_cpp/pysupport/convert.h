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

#ifndef pysupport_convert
#define pysupport_convert

#include <map>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#include "Python.h"

namespace pysupport {

class PyTmpRef {
    // RAII for a Python object reference.  Treat like a PyObject*
    // and don't worry about the refernce count being decremented.
    // Not appropriate for borrowed references (just use PyObject*).
    PyObject* obj;
public:
    PyTmpRef(): obj(nullptr) {}
    PyTmpRef(PyObject* o): obj(o) {}
    ~PyTmpRef() { Py_CLEAR(obj); }
    operator PyObject*() const { return obj; }
    operator PyVarObject*() const { return reinterpret_cast<PyVarObject*>(obj); }
    PyTmpRef& operator=(PyObject* o) noexcept {
        if (obj != o) {
            Py_CLEAR(obj);
            obj = o;
        }
        return *this;
    }
    operator bool() noexcept {
        return obj != nullptr;
    }
};

inline std::string _make_msg(std::initializer_list<const char*> parts) {
    std::stringstream msg;
    for (auto part: parts)
        msg << part;
    return msg.str();
}

class PySupportError : public std::invalid_argument {
public:
    PySupportError(const std::string& msg) : std::invalid_argument(msg) {}
};

class ErrStringCreate : public PySupportError {
public:
    ErrStringCreate(const char* item_description) :
        PySupportError(_make_msg({"Can't create Python string for ", item_description})) {}
};

class ErrLongCreate : public PySupportError {
public:
    ErrLongCreate(const char* item_description) :
        PySupportError(_make_msg({"Can't create Python int for ", item_description})) {}
};

class ErrListCreate : public PySupportError {
public:
    ErrListCreate(const char* item_description) :
        PySupportError(_make_msg({"Can't allocate Python list of ", item_description})) {}
};

class ErrSetCreate : public PySupportError {
public:
    ErrSetCreate(const char* item_description) :
        PySupportError(_make_msg({"Can't allocate Python set of ", item_description, "s"})) {}
};

class ErrDictCreate : public PySupportError {
public:
    ErrDictCreate(const char* key_description, const char* val_description) :
        PySupportError(_make_msg({"Can't allocate Python dict for (", key_description,
            ") -> (list of ", val_description, "s) mapping"})) {}
};

class ErrNotList : public PySupportError {
public:
    ErrNotList(const char* item_description) :
        PySupportError(_make_msg({item_description, " info is not a list"})) {}
};

class ErrListItemNotString : public PySupportError {
public:
    ErrListItemNotString(const char* item_description) :
        PySupportError(_make_msg({item_description, " is not a Unicode string"})) {}
};

class ErrListItemNotInt : public PySupportError {
public:
    ErrListItemNotInt(const char* item_description) :
        PySupportError(_make_msg({item_description, " is not an integer"})) {}
};

class ErrListItemNotFloat : public PySupportError {
public:
    ErrListItemNotFloat(const char* item_description) :
        PySupportError(_make_msg({item_description, " is not a float"})) {}
};

template <class Str>
PyObject* cchar_to_pystring(Str& cchar, const char* item_description) {
    auto pyobj = PyUnicode_DecodeUTF8(cchar.c_str(), cchar.size(), "replace");
    if (pyobj == nullptr)
        throw ErrStringCreate(item_description);
    return pyobj;
}

template <class Ptr>
PyObject* cptr_to_pyint(Ptr ptr, const char* item_description) {
    auto pyobj = PyLong_FromVoidPtr(static_cast<void*>(ptr));
    if (pyobj == nullptr)
        throw ErrLongCreate(item_description);
    return pyobj;
}

template <class Int>
PyObject* cint_to_pyint(Int integer, const char* item_description) {
    auto pyobj = PyLong_FromLong(static_cast<long>(integer));
    if (pyobj == nullptr)
        throw ErrLongCreate(item_description);
    return pyobj;
}

template <class Str>
PyObject* cvec_of_char_to_pylist(std::vector<Str>& vec, const char* item_description) {
    PyObject* pylist = PyList_New(vec.size());
    if (pylist == nullptr)
        throw ErrListCreate(item_description);
    typename std::vector<Str>::size_type i = 0;
    for (auto& c_item: vec) {
        PyList_SET_ITEM(pylist, i++, cchar_to_pystring(c_item, item_description));
    }
    return pylist;
}

inline
PyObject* cvec_of_cvec_of_char_to_pylist(std::vector<std::vector<char>>& vec,
    const char* item_description)
{
    PyObject* pylist = PyList_New(vec.size());
    if (pylist == nullptr)
        throw ErrListCreate(item_description);
    typename std::vector<std::vector<char>>::size_type i = 0;
    for (auto& c_item: vec) {
        auto str = std::string(c_item.begin(), c_item.end());
        PyList_SET_ITEM(pylist, i++, cchar_to_pystring(str, item_description));
    }
    return pylist;
}

template <class Int>
PyObject* cvec_of_int_to_pylist(std::vector<Int>& vec, const char* item_description) {
    PyObject* pylist = PyList_New(vec.size());
    if (pylist == nullptr)
        throw ErrListCreate(item_description);
    typename std::vector<Int>::size_type i = 0;
    for (auto& c_item: vec) {
        PyList_SET_ITEM(pylist, i++, PyLong_FromLong(c_item));
    }
    return pylist;
}

template <class Ptr>
PyObject* cvec_of_ptr_to_pylist(std::vector<Ptr>& vec, const char* item_description) {
    PyObject* pylist = PyList_New(vec.size());
    if (pylist == nullptr)
        throw ErrListCreate(item_description);
    typename std::vector<Ptr>::size_type i = 0;
    for (auto& c_item: vec) {
        PyList_SET_ITEM(pylist, i++,
            PyLong_FromVoidPtr(const_cast<void*>(static_cast<const void*>(c_item))));
    }
    return pylist;
}

template <class Set>
PyObject* cset_of_chars_to_pyset(Set& cset, const char* item_description) {
    PyObject* pyset = PySet_New(nullptr);
    if (pyset == nullptr)
        throw ErrSetCreate(item_description);
    for (auto& c_item: cset)
        PySet_Add(pyset, cchar_to_pystring(c_item, item_description));
    return pyset;
}

template <class Map>
PyObject* cmap_of_chars_to_pydict(Map& cmap,
        const char* key_description, const char* val_description) {
    PyObject* pydict = PyDict_New();
    if (pydict == nullptr)
        throw ErrDictCreate(key_description, val_description);
    for (auto key_val: cmap) {
        PyObject* pykey = cchar_to_pystring(key_val.first, key_description);
        PyObject* pyval = cvec_of_char_to_pylist(key_val.second, val_description);
        PyDict_SetItem(pydict, pykey, pyval);
        Py_DECREF(pykey);
        Py_DECREF(pyval);
    }
    return pydict;
}

template <class Ptr, class Int>
PyObject* cmap_of_ptr_int_to_pydict(const std::map<Ptr, Int>& cmap,
        const char* key_description, const char* val_description) {
    PyObject* pydict = PyDict_New();
    if (pydict == nullptr)
        throw ErrDictCreate(key_description, val_description);
    for (auto key_val: cmap) {
        PyObject* pykey = cptr_to_pyint(key_val.first, key_description);
        PyObject* pyval = cint_to_pyint(key_val.second, val_description);
        PyDict_SetItem(pydict, pykey, pyval);
        Py_DECREF(pykey);
        Py_DECREF(pyval);
    }
    return pydict;
}

inline const char* pystring_to_cchar(PyObject* string, const char* item_description) {
    if (!PyUnicode_Check(string))
        throw ErrListItemNotString(item_description);
    return PyUnicode_AsUTF8(string);
}

inline long pyint_to_clong(PyObject* pyint, const char* item_description) {
    if (!PyLong_Check(pyint))
        throw ErrListItemNotInt(item_description);
    return PyLong_AsLong(pyint);
}

inline double pyfloat_to_cdouble(PyObject* pyfloat, const char* item_description) {
    if (!PyFloat_Check(pyfloat))
        throw ErrListItemNotFloat(item_description);
    return PyFloat_AS_DOUBLE(pyfloat);
}

template <class Contained>
void pylist_of_string_to_cvec(PyObject* pylist, std::vector<Contained>& cvec,
        const char* item_description) {
    if (!PyList_Check(pylist))
        throw ErrNotList(item_description);
    auto num_items = PyList_GET_SIZE(pylist);
    for (decltype(num_items) i = 0; i < num_items; ++i) {
        PyObject* item = PyList_GET_ITEM(pylist, i);
        cvec.emplace_back(pystring_to_cchar(item, item_description));
    }
}

template <class Contained>
void pysequence_of_string_to_cvec(PyObject* pylist, std::vector<Contained>& cvec,
        const char* item_description) {
    PyTmpRef seq = PySequence_Fast(pylist, "not a sequence");
    if (!seq) {
        PyErr_Clear();
        throw ErrNotList(item_description);
    }
    auto num_items = PySequence_Fast_GET_SIZE(pylist);
    for (decltype(num_items) i = 0; i < num_items; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(pylist, i);
        cvec.emplace_back(pystring_to_cchar(item, item_description));
    }
}

inline void pylist_of_string_to_cvec_of_cvec(PyObject* pylist,
        std::vector<std::vector<char>>& cvec, const char* item_description)
{
    if (!PyList_Check(pylist))
        throw ErrNotList(item_description);
    auto num_items = PyList_GET_SIZE(pylist);
    cvec.resize(num_items);
    for (decltype(num_items) i = 0; i < num_items; ++i) {
        PyObject* item = PyList_GET_ITEM(pylist, i);
        auto& char_vec = cvec[i];
        for (auto c = pystring_to_cchar(item, item_description); *c != '\0'; ++c)
            char_vec.push_back(*c);
    }
}

template <class Int>
void pylist_of_int_to_cvec(PyObject* pylist, std::vector<Int>& cvec, const char* item_description) {
    if (!PyList_Check(pylist))
        throw ErrNotList(item_description);
    auto num_items = PyList_GET_SIZE(pylist);
    cvec.resize(num_items);
    for (decltype(num_items) i = 0; i < num_items; ++i) {
        PyObject* item = PyList_GET_ITEM(pylist, i);
        cvec[i] = static_cast<Int>(pyint_to_clong(item, item_description));
    }
}

template <class Float>
void pylist_of_float_to_cvec(PyObject* pylist, std::vector<Float>& cvec,
        const char* item_description) {
    if (!PyList_Check(pylist))
        throw ErrNotList(item_description);
    auto num_items = PyList_GET_SIZE(pylist);
    cvec.resize(num_items);
    for (decltype(num_items) i = 0; i < num_items; ++i) {
        PyObject* item = PyList_GET_ITEM(pylist, i);
        cvec[i] = static_cast<Float>(pyfloat_to_cdouble(item, item_description));
    }
}

} //  namespace pysupport

#endif  // pysupport_convert
