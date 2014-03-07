// vim: set expandtab ts=4 sw=4:
#ifndef blob_StructBlob
#define blob_StructBlob

#include "Blob.h"

extern PyTypeObject StructBlob_type;

#include "atomstruct/AtomicStructure.h"
typedef UniqueBlob<AtomicStructure> StructBlob;

extern template BLOB_IMEX PyObject* newBlob<StructBlob>(PyTypeObject*);

#endif  // blob_StructBlob
