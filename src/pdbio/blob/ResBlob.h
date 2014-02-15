// vim: set expandtab ts=4 sw=4:
#ifndef blob_ResBlob
#define blob_ResBlob

#include "Blob.h"

extern PyTypeObject ResBlob_type;

class Residue;
typedef RawBlob<Residue> ResBlob;

extern template BLOB_IMEX PyObject* newBlob<ResBlob>(PyTypeObject*);

#endif  // blob_ResBlob
