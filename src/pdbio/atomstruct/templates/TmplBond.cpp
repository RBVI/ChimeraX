// vim: set expandtab ts=4 sw=4:
#include "restmpl.h"

TmplBond::TmplBond(TmplMolecule *, TmplAtom *a0, TmplAtom *a1)
{
    _atoms[0] = a0;
    _atoms[1] = a1;
    a0->add_bond(this);
    a1->add_bond(this);
}

TmplAtom *
TmplBond::other_atom(const TmplAtom *a) const {
    if (a == _atoms[0])
        return _atoms[1];
    if (a == _atoms[1])
        return _atoms[0];
    return NULL;
}

TmplBond::~TmplBond()
{
}

