#ifndef templates_TmplAtom
#define	templates_TmplAtom

#include <map>
#include <vector>
#include "TmplCoordSet.h"
#include "../Element.h"


class TmplMolecule;
class TmplResidue;
class TmplBond;

class TmplAtom {
	friend class TmplMolecule;
	friend class TmplResidue;
	void	operator=(const TmplAtom &);	// disable
		TmplAtom(const TmplAtom &);	// disable
		~TmplAtom();
	TmplMolecule	*_molecule;
	TmplResidue	*_residue;
public:
	void	add_bond(TmplBond *b);
	typedef std::map<TmplAtom*, TmplBond *> BondsMap;
	const BondsMap	&bonds_map() const { return _bonds; }
	TmplMolecule	*molecule() const { return _molecule; }
	TmplResidue	*residue() const { return _residue; }
	std::string		name() const { return _name; }
private:
	std::string	_name;
	Element	_element;
	BondsMap	_bonds;
public:
	static const unsigned int COORD_UNASSIGNED = ~0u;
	void		set_coord(const TmplCoord &c);
	void		set_coord(const TmplCoord &c, TmplCoordSet *cs);
private:
	mutable unsigned int _index;
	int	new_coord(const TmplCoord &c) const;
private:
	std::string     _idatm_type;
public:
	std::string  idatm_type() const { return _idatm_type; }
	void	set_idatm_type(const char *i) { _idatm_type = i; }
private:
	TmplAtom(TmplMolecule *, std::string &n, Element e);
};

#endif  // templates_TmplAtom
