// vi: set expandtab ts=4 sw=4:
#ifndef blob_blob
#define blob_blob

#include <Python.h>
#include <vector>
#include <memory>
#include "imex.h"

namespace blob {
    
template <class MolItem>
struct Blob: public PyObject {
public:
    typedef MolItem  MolType;
    PyObject* _weaklist;
};

template <class MolItem>
class SharedAPIPointer {
    MolItem* _ptr;
public:
    SharedAPIPointer(MolItem *ptr): _ptr(ptr) {}
    MolItem* operator->() const { return _ptr; }
    MolItem* get() const { return _ptr; }
};

template <class MolItem>
class SharedBlob: public Blob<MolItem> {
public:
    typedef std::vector<std::shared_ptr<MolItem>>  ItemsType;
    ItemsType*  _items;
};

template <class MolItem>
class RawBlob: public Blob<MolItem> {
public:
    typedef std::vector<SharedAPIPointer<MolItem>>  ItemsType;
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
