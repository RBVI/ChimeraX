// vi: set expandtab ts=4 sw=4:
#ifndef blob_PseudoBlob
#define blob_PseudoBlob

#include "Blob.h"
#include <atomstruct/Pseudobond.h>

namespace blob {
    
extern PyTypeObject PseudoBlob_type;

using atomstruct::PBond;
typedef Blob<PBond, SharedAPIPointer<PBond>> PseudoBlob;

extern template BLOB_IMEX PyObject* new_blob<PseudoBlob>(PyTypeObject*);

}  // namespace blob

#endif  // blob_PseudoBlob
