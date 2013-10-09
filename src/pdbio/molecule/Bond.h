#ifndef molecule_Bond
#define molecule_Bond

#include "base-geom/Real.h"

class Atom;
class Molecule;

class Bond {
	friend class Molecule;
public:
	typedef Atom *  Atoms[2];

private:
	Bond(Molecule *, Atom *, Atom *);
	Atoms  _atoms;

public:
	const Atoms	&  atoms() const { return _atoms; }
	Atom *  other_atom(Atom *a) const;
	Real  sqlength() const;

};
#endif  // molecule_Bond
