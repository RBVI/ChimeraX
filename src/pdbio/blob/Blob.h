// vim: set expandtab ts=4 sw=4:
#ifndef blob_blob
#define blob_blob

#include <Python.h>
#include <vector>
#include <memory>
#include "imex.h"

namespace blob {
    
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
    ItemsType*  _items;
};

template <class MolItem>
class RawBlob: public Blob {
public:
    typedef std::vector<UniqueAPIPointer<MolItem>>  ItemsType;
    ItemsType*  _items;
};

template <class BlobType>
PyObject*
newBlob(PyTypeObject* type)
{
    BlobType* self;
    self = static_cast<BlobType*>(type->tp_alloc(type, 0));
    if (self != NULL) {
        self->_items = new typename BlobType::ItemsType;
    }
    return static_cast<PyObject*>(self);
}

}  // namespace blob

#endif  // blob_blob
