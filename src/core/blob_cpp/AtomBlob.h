// vi: set expandtab ts=4 sw=4:
#ifndef blob_AtomBlob
#define blob_AtomBlob

#include "Blob.h"
#include <atomstruct/Atom.h>

namespace blob {
    
extern PyTypeObject AtomBlob_type;

using atomstruct::Atom;
typedef Blob<Atom, SharedAPIPointer<Atom>> AtomBlob;

extern template BLOB_IMEX PyObject* new_blob<AtomBlob>(PyTypeObject*);

}  // namespace blob

#endif  // blob_AtomBlob
