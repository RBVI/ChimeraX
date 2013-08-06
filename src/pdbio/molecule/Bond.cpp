#include "Bond.h"
#include "Atom.h"
#include <stdexcept>

Bond::Bond(Molecule *m, Atom *a1, Atom *a2)
{
	if (a1 == a2)
		throw std::invalid_argument("Attempt to bond atom with itself.");
	if (a1->molecule() != m || a2->molecule() != m)
		throw std::invalid_argument("Cannot add bond to molecule involving atoms from"
			" other molecule(s).");
	if (a1->connects_to(a2))
		throw std::invalid_argument("Attempt to form duplicate covalent bond.");
	_atoms[0] = a1;
	_atoms[1] = a2;
	a1->add_bond(this);
	a2->add_bond(this);
}

Atom *
Bond::other_atom(Atom *a) const
{
	if (a == _atoms[0])
		return _atoms[1];
	if (a == _atoms[1])
		return _atoms[0];
	throw std::invalid_argument("Atom argument to other_atom() not part of bond.");
}

Real
Bond::sqlength() const
{
	return _atoms[0]->coord().sqdistance(_atoms[1]->coord());
}
