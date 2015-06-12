// vi: set expandtab ts=4 sw=4:
#ifndef blob_blob
#define blob_blob

#include <memory>
#include <Python.h>
#include <vector>

#include <basegeom/destruct.h>
#include "imex.h"

namespace blob {
    
template <class ItemsType>
class BlobObserver: public basegeom::DestructionObserver {
private:
    ItemsType*  _items;
public:
    BlobObserver(ItemsType* items): _items(items) {}
    void  destructors_done(const std::set<void*>& destroyed) {
        ItemsType remaining;
        for (auto i: *_items) {
            if (destroyed.find((void*)(i.get())) == destroyed.end())
                remaining.push_back(i);
        }
        _items->swap(remaining);
    }
};

template <class MolItem, class PtrClass>
class Blob: public PyObject, public basegeom::DestructionObserver {
public:
    typedef MolItem  MolType;
    PyObject*  _weaklist;
    typedef std::vector<PtrClass>  ItemsType;
    ItemsType*  _items;
    BlobObserver<ItemsType>*  _observer;
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
        self->_observer =
            new BlobObserver<typename BlobType::ItemsType>(self->_items);
    }
    return static_cast<PyObject*>(self);
}

template <class BlobType>
PyObject*
PyType_NewBlob(PyTypeObject* type, PyObject*, PyObject*)
{
    return new_blob<BlobType>(type);
}

inline bool
init_structaccess()
{
    return PyImport_ImportModule("chimera.core.structaccess") != nullptr;
}

}  // namespace blob

#endif  // blob_blob
