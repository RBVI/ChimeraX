#ifndef molecule_Bond
#define molecule_Bond

#include "base-geom/Real.h"
#include "base-geom/Connection.h"

class Atom;
class Molecule;

class Bond: public Connection<Atom, Bond> {
	friend class Molecule;
public:
	typedef End_points  Atoms;

private:
	Bond(Molecule *, Atom *, Atom *);

public:
	const Atoms	&  atoms() const { return end_points(); }
	Atom *  other_atom(Atom *a) const { return other_end(a); }
};
#endif  // molecule_Bond
