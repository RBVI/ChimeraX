// vi: set expandtab ts=4 sw=4:
#ifndef blob_StructBlob
#define blob_StructBlob

#include "Blob.h"
#include <atomstruct/AtomicStructure.h>

namespace blob {
    
extern PyTypeObject StructBlob_type;

using atomstruct::AtomicStructure;
typedef Blob<AtomicStructure, std::shared_ptr<AtomicStructure>> StructBlob;

extern template BLOB_IMEX PyObject* new_blob<StructBlob>(PyTypeObject*);

}  // namespace blob

#endif  // blob_StructBlob
