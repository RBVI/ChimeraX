// vim: set expandtab ts=4 sw=4:
#ifndef access_AtomBlob
#define access_AtomBlob

#include "Blob.h"

extern PyTypeObject AtomBlob_type;

#include "molecule/Atom.h"
typedef RawBlob<Atom> AtomBlob;

extern template BLOB_IMEX PyObject* newBlob<AtomBlob>(PyTypeObject*);

#endif  // access_AtomBlob
