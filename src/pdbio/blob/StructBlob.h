// vim: set expandtab ts=4 sw=4:
#ifndef blob_StructBlob
#define blob_StructBlob

#include "Blob.h"
#include "atomstruct/AtomicStructure.h"

namespace blob {
    
extern PyTypeObject StructBlob_type;

typedef SharedBlob<atomstruct::AtomicStructure> StructBlob;

extern template BLOB_IMEX PyObject* newBlob<StructBlob>(PyTypeObject*);

}  // namespace blob

#endif  // blob_StructBlob
