// vim: set expandtab ts=4 sw=4:
#ifndef access_MolBlob
#define access_MolBlob

#include "Blob.h"

extern PyTypeObject MolBlob_type;

#include "molecule/Molecule.h"
typedef UniqueBlob<Molecule> MolBlob;

extern template BLOB_IMEX PyObject* newBlob<MolBlob>(PyTypeObject*);

#endif  // access_MolBlob
