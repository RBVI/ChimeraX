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

#define ATOMSTRUCT_EXPORT
#include "restmpl.h"

namespace tmpl {

Bond::Bond(Molecule *, Atom *a0, Atom *a1)
{
    _atoms[0] = a0;
    _atoms[1] = a1;
    a0->add_bond(this);
    a1->add_bond(this);
}

Atom *
Bond::other_atom(const Atom *a) const {
    if (a == _atoms[0])
        return _atoms[1];
    if (a == _atoms[1])
        return _atoms[0];
    return NULL;
}

Bond::~Bond()
{
}

}  // namespace tmpl
