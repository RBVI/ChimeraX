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

#ifndef templates_resinternal
#define templates_resinternal

#include <map>
#include <string>

#include "restmpl.h"
#include "../string_types.h"

namespace tmpl {

using atomstruct::ResName;

struct ResInit {
    Residue    *(*start)(Molecule *);
    Residue    *(*middle)(Molecule *);
    Residue    *(*end)(Molecule *);
    ResInit(): start(0), middle(0), end(0) {}
};

typedef std::map<ResName, ResInit> ResInitMap;

}  // namespace tmpl

#endif  // templates_resinternal
