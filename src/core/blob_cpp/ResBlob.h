// vi: set expandtab ts=4 sw=4:
#ifndef blob_ResBlob
#define blob_ResBlob

#include "Blob.h"
#include <atomstruct/Residue.h>

namespace blob {
    
extern PyTypeObject ResBlob_type;

using atomstruct::Residue;
typedef Blob<Residue, SharedAPIPointer<Residue>> ResBlob;

extern template BLOB_IMEX PyObject* new_blob<ResBlob>(PyTypeObject*);

}  // namespace blob

#endif  // blob_ResBlob
