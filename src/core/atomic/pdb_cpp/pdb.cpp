// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2016 Regents of the University of California.
 * All rights reserved.  This software provided pursuant to a
 * license agreement containing restrictions on its disclosure,
 * duplication and use.  For details see:
 * http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
 * This notice must be embedded in or attached to all copies,
 * including partial copies, of the software or any revisions
 * or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#include "PDB.h"
#include <string.h>

namespace pdb {

void
PDB::set_type(RecordType t)
{
    if (t == UNKNOWN) {
        // optimize default case (skip memset())
        r_type = t;
        unknown.junk[0] = '\0';
        return;
    }
    memset(this, 0, sizeof *this);
    r_type = t;
    switch (t) {
      default:
        break;
      case ATOM:
        atom.occupancy = 1.0;
        break;
    }
}

int
PDB::byte_cmp(const PDB &l, const PDB &r)
{
    return memcmp(&l, &r, sizeof (PDB));
}

void
PDB::reset_state()
{
    input_version = 0;
    atom_serial_number = 10000;
    sigatm_serial_number = 10000;
}

}  // namespace pdb
