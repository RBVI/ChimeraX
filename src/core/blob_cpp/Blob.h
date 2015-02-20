// vi: set expandtab ts=4 sw=4:
#ifndef blob_blob
#define blob_blob

#include <Python.h>
#include <vector>
#include <memory>
#include "imex.h"

namespace blob {
    
template <class MolItem, class PtrClass>
struct Blob: public PyObject {
public:
    typedef MolItem  MolType;
    PyObject* _weaklist;
    typedef std::vector<PtrClass>  ItemsType;
    ItemsType*  _items;
};

template <class MolItem>
class SharedAPIPointer {
    MolItem* _ptr;
public:
    SharedAPIPointer(MolItem *ptr): _ptr(ptr) {}
    MolItem* operator->() const { return _ptr; }
    MolItem* get() const { return _ptr; }
};

template <class BlobType>
PyObject*
new_blob(PyTypeObject* type)
{
    BlobType* self;
    self = static_cast<BlobType*>(type->tp_alloc(type, 0));
    if (self != NULL) {
        self->_items = new typename BlobType::ItemsType;
    }
    return static_cast<PyObject*>(self);
}

template <class BlobType>
PyObject*
PyType_NewBlob(PyTypeObject* type, PyObject*, PyObject*)
{
    return new_blob<BlobType>(type);
}

}  // namespace blob

#endif  // blob_blob
