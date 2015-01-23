// vi: set expandtab ts=4 sw=4:
#ifndef blob_BondBlob
#define blob_BondBlob

#include "Blob.h"
#include <atomstruct/Bond.h>

namespace blob {
    
extern PyTypeObject BondBlob_type;

typedef RawBlob<atomstruct::Bond> BondBlob;

extern template BLOB_IMEX PyObject* newBlob<BondBlob>(PyTypeObject*);

}  // namespace blob

#endif  // blob_BondBlob
