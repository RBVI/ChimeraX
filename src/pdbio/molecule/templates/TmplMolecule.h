#ifndef templates_TmplMolecule
#define	templates_TmplMolecule

#include <set>
#include <map>
#include <vector>
#include <string>
#include "TAexcept.h"
#include "TmplAtom.h"
#include "TmplBond.h"
#include "TmplCoordSet.h"
#include "TmplResidue.h"

class TmplMolecule {
public:
		~TmplMolecule();
	TmplAtom	*new_atom(std::string n, Element e);
	typedef std::set<TmplAtom *> Atoms;
	typedef std::set<TmplBond *> Bonds;
	typedef std::vector<TmplCoordSet *> CoordSets;
	typedef std::map<std::string, TmplResidue *> Residues;
private:
	Atoms	_atoms;
	Bonds	_bonds;
	CoordSets	_coord_sets;
	Residues	_residues;
public:
#ifdef UNPORTED
	void	deleteAtom(TmplAtom *element);
	inline const Atoms	&atoms() const;
#endif  // UNPORTED
	TmplBond	*new_bond(TmplAtom *a0, TmplAtom *a1);
#ifdef UNPORTED
	TmplBond	*newBond(TmplAtom *a[2]);
	void	deleteBond(TmplBond *element);
	inline const Bonds	&bonds() const;
#endif  // UNPORTED
	TmplCoordSet	*new_coord_set(int key);
#ifdef UNPORTED
	TmplCoordSet	*newCoordSet(int key, int size);
	void	deleteCoordSet(TmplCoordSet *element);
#endif  // UNPORTED
	const CoordSets	&coord_sets() const { return _coord_sets; }
	TmplCoordSet	*find_coord_set(int) const;
	TmplResidue	*new_residue(const char *t);
#ifdef UNPORTED
	TmplResidue	*newResidue(Symbol t, Symbol chain, int pos, char insert);
	void	deleteResidue(TmplResidue *element);
	inline Residues residues() const;
	typedef std::map<Symbol, TmplResidue *> ResiduesMap;
	inline const ResiduesMap	&residuesMap() const;
	typedef std::vector<Symbol> ResidueKeys;
	inline ResidueKeys	residueNames() const;
#endif  // UNPORTED
	TmplResidue	*find_residue(const std::string &) const;
	void		set_active_coord_set(TmplCoordSet *cs);
	TmplCoordSet	*active_coord_set() const { return _active_cs; }
private:
	TmplCoordSet	*_active_cs;
public:
	TmplMolecule();
};

#endif  // templates_TmplMolecule
