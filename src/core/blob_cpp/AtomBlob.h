// vi: set expandtab ts=4 sw=4:
#ifndef blob_AtomBlob
#define blob_AtomBlob

#include "Blob.h"
#include <atomstruct/Atom.h>

namespace blob {
    
extern PyTypeObject AtomBlob_type;

typedef RawBlob<atomstruct::Atom> AtomBlob;

extern template BLOB_IMEX PyObject* newBlob<AtomBlob>(PyTypeObject*);

}  // namespace blob

#endif  // blob_AtomBlob
