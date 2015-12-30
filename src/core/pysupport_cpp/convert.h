// vi: set expandtab ts=4 sw=4:
#ifndef pysupport_convert
#define pysupport_convert

#include <sstream>
#include <string>
#include <vector>

#include "Python.h"

namespace pysupport {

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

class ErrListCreate : public PySupportError {
public:
    ErrListCreate(const char* item_description) :
        PySupportError(_make_msg({"Can't allocate Python list of ", item_description})) {}
};

class ErrDictCreate : public PySupportError {
public:
    ErrDictCreate(const char* key_description, const char* val_description) :
        PySupportError(_make_msg({"Can't allocate Python dict for (", key_description,
            ") -> (list of ", val_description, ") mapping"})) {}
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

template <class Str>
PyObject* cchar_to_pystring(Str& cchar, const char* item_description) {
    auto pyobj = PyUnicode_FromString(cchar.c_str());
    if (pyobj == nullptr)
        throw ErrStringCreate(item_description);
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

inline char* pystring_to_cchar(PyObject* string, const char* item_description) {
    if (!PyUnicode_Check(string))
        throw ErrListItemNotString(item_description);
    return PyUnicode_AsUTF8(string);
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

} //  namespace pysupport

#endif  // pysupport_convert
