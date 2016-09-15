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

namespace pdb {

std::istream &
operator>>(std::istream &s, PDB &p)
{
    char    buf[4 * PDB::BUF_LEN];

    s.getline(buf, 4 * PDB::BUF_LEN);
    p = PDB(buf);
    return s;
}

}  // namespace pdb
