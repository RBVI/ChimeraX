#ifndef templates_TmplBond
#define	templates_TmplBond

#ifdef UNPORTED
#include "TAexcept.h"
#include <_chimera/Array.h>		// use chimera::Array
#include <_chimera/Geom3d.h>		// use chimera::Real
#endif  // UNPORTED

class TmplAtom;
class TmplMolecule;
#ifdef UNPORTED
using chimera::Array;
using chimera::Real;
#endif  // UNPORTED

class TmplBond {
	friend class TmplAtom;
	friend class TmplMolecule;
	void	operator=(const TmplBond &);	// disable
		TmplBond(const TmplBond &);	// disable
		~TmplBond();
public:
	typedef TmplAtom *	Atoms[2];
private:
	Atoms	_atoms;
public:
	const Atoms	&atoms() const { return _atoms; }
#ifdef UNPORTED
	TmplAtom	*findAtom(size_t) const;
#endif  // UNPORTED
	TmplAtom		*other_atom(const TmplAtom *a) const;
#ifdef UNPORTED
	inline Real		length() const;
	inline Real		sqlength() const;
#endif  // UNPORTED
private:
	TmplBond(TmplMolecule *, TmplAtom *a0, TmplAtom *a1);
#ifdef UNPORTED
	TmplBond(TmplMolecule *, TmplAtom *a[2]);
#endif  // UNPORTED
};

#endif  // templates_TmplBond
