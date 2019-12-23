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

#include <pyinstance/PythonInstance.declare.h>
#include "../imex.h"
#include "../Real.h"

namespace tmpl {

class Atom;
class Molecule;

class ATOMSTRUCT_IMEX Bond: public pyinstance::PythonInstance<Bond> {
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
    const Atoms&  atoms() const { return _atoms; }
    atomstruct::Real  length() const;
    Atom*  other_atom(const Atom *a) const;
private:
    Bond(Molecule *, Atom *a0, Atom *a1);
};

}  // namespace tmpl

#include "Atom.h"

namespace tmpl {
    
inline atomstruct::Real
Bond::length() const {
    return _atoms[0]->coord().distance(_atoms[1]->coord());
}

}

#endif  // templates_Bond
