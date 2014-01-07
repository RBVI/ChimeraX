// vim: set expandtab ts=4 sw=4:
#ifndef access_blob
#define access_blob

#include "Python.h"
#include <vector>
#include <memory>

class Blob: public PyObject {
public:
    PyObject* _weaklist;
};

template <class MolItem>
class UniqueAPIPointer {
    MolItem* _ptr;
public:
    UniqueAPIPointer(MolItem *ptr): _ptr(ptr) {}
    MolItem* operator->() const { return _ptr; }
    MolItem* get() const { return _ptr; }
};

template <class MolItem>
class UniqueBlob: public Blob {
public:
    typedef std::vector<std::unique_ptr<MolItem>>  ItemsType;
    ItemsType  _items;
};

template <class MolItem>
class RawBlob: public Blob {
public:
    typedef std::vector<UniqueAPIPointer<MolItem>>  ItemsType;
    ItemsType  _items;
};

#endif  // access_blob
