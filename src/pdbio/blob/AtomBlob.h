// vim: set expandtab ts=4 sw=4:
#ifndef blob_AtomBlob
#define blob_AtomBlob

#include "Blob.h"

extern PyTypeObject AtomBlob_type;

#include "atomstruct/Atom.h"
typedef RawBlob<Atom> AtomBlob;

extern template BLOB_IMEX PyObject* newBlob<AtomBlob>(PyTypeObject*);

#endif  // blob_AtomBlob
