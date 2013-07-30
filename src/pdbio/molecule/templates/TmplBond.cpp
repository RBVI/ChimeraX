#include "restmpl.h"

#ifdef UNPORTED
TmplAtom *
TmplBond::findAtom(std::size_t index) const
{
	if (index >= Atoms_.size())
		throw std::out_of_range("index out of range");
	return Atoms_[index];
}
#endif  // UNPORTED
TmplBond::TmplBond(TmplMolecule *, TmplAtom *a0, TmplAtom *a1)
{
	_atoms[0] = a0;
	_atoms[1] = a1;
	a0->add_bond(this);
	a1->add_bond(this);
}
#ifdef UNPORTED
TmplBond::TmplBond(TmplMolecule *, TmplAtom *a[2])
{
	Atoms_[0] = a[0];
	Atoms_[1] = a[1];
	a[0]->addBond(this);
	a[1]->addBond(this);
}
#endif  // UNPORTED

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

