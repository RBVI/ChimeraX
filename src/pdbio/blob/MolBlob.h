// vim: set expandtab ts=4 sw=4:
#ifndef blob_MolBlob
#define blob_MolBlob

#include "Blob.h"

extern PyTypeObject MolBlob_type;

#include "molecule/Molecule.h"
typedef UniqueBlob<Molecule> MolBlob;

extern template BLOB_IMEX PyObject* newBlob<MolBlob>(PyTypeObject*);

#endif  // blob_MolBlob
