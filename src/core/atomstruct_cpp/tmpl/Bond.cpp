// vi: set expandtab ts=4 sw=4:

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
