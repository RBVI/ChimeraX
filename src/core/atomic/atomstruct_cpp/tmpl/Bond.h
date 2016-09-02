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

#ifndef templates_Bond
#define    templates_Bond

#include "../imex.h"

namespace tmpl {

class Atom;
class Molecule;

class ATOMSTRUCT_IMEX Bond {
    friend class Atom;
    friend class Molecule;
    void    operator=(const Bond &);    // disable
        Bond(const Bond &);    // disable
        ~Bond();
public:
    typedef Atom *    Atoms[2];
private:
    Atoms    _atoms;
public:
    const Atoms    &atoms() const { return _atoms; }
    Atom        *other_atom(const Atom *a) const;
private:
    Bond(Molecule *, Atom *a0, Atom *a1);
};

}  // namespace tmpl

#endif  // templates_Bond
