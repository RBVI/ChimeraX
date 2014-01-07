// vim: set expandtab ts=4 sw=4:
#ifndef access_MolBlob
#define access_MolBlob

#include "Blob.h"

extern PyTypeObject MolBlob_type;

#include "molecule/Molecule.h"
typedef UniqueBlob<Molecule> MolBlob;

#endif  // access_MolBlob
