// vi: set expandtab ts=4 sw=4:
#ifndef pysupport_convert
#define pysupport_convert

#include <sstream>
#include <string>
#include <vector>

#include "Python.h"

namespace pysupport {

inline std::string _make_msg(const char* item_description, const char* trailer) {
    std::stringstream msg;
    msg << item_description << trailer;
    return msg.str();
}

class PySupportError : public std::invalid_argument {
public:
    PySupportError(const std::string& msg) : std::invalid_argument(msg) {}
};

class ErrNotList : public PySupportError {
public:
    ErrNotList(const char* item_description) :
        PySupportError(_make_msg(item_description, " info is not a list")) {}
};

class ErrListItemNotString : public PySupportError {
public:
    ErrListItemNotString(const char* item_description) :
        PySupportError(_make_msg(item_description, " is not a Unicode string")) {}
};

inline char* pystring_to_cchar(PyObject* string, const char* item_description) {
    if (!PyUnicode_Check(string))
        throw ErrListItemNotString(item_description);
    return PyUnicode_AsUTF8(string);
}

template <class Contained>
void pylist_of_string_to_cvector(PyObject* pylist, std::vector<Contained>& cvector,
        const char* item_description) {
    if (!PyList_Check(pylist))
        throw ErrNotList(item_description);
    auto num_items = PyList_GET_SIZE(pylist);
    for (decltype(num_items) i = 0; i < num_items; ++i) {
        PyObject* item = PyList_GET_ITEM(pylist, i);
        cvector.emplace_back(pystring_to_cchar(item, item_description));
    }
}

} //  namespace pysupport

#endif  // pysupport_convert
