#ifndef templates_TmplAtom
#define	templates_TmplAtom

#include <map>
#include <vector>
#include "TmplCoordSet.h"
#ifdef UNPORTED
#include "TAexcept.h"
#endif  // UNPORTED
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
#ifdef UNPORTED
	void	removeBond(TmplBond *element);
	typedef std::vector<TmplBond *> Bonds;
	inline Bonds bonds() const;
#endif  // UNPORTED
	typedef std::map<TmplAtom*, TmplBond *> BondsMap;
	const BondsMap	&bonds_map() const { return _bonds; }
#ifdef UNPORTED
	typedef std::vector<TmplAtom*> BondKeys;
	inline BondKeys	neighbors() const;
	TmplBond	*findBond(TmplAtom*) const;
#endif  // UNPORTED
	TmplMolecule	*molecule() const { return _molecule; }
	TmplResidue	*residue() const { return _residue; }
#ifdef UNPORTED
	inline TmplBond		*connectsTo(TmplAtom *a) const;
#endif  // UNPORTED
	std::string		name() const { return _name; }
#ifdef UNPORTED
	inline void		setName(Symbol s);
	inline Element	element() const;
	void		setElement(Element e);
	// Atom_idatm overrides setElement, so using "old-fashioned" inlining
	// to work around the fact that wrappy isn't smart enough not to
	// elide the following implementations
	void		setElement(int e) { setElement(Element(e)); };
	void		setElement(const char *e) { setElement(Element(e)); };


#endif  // UNPORTED
private:
	std::string	_name;
	Element	_element;
	BondsMap	_bonds;
public:
	static const unsigned int COORD_UNASSIGNED = ~0u;
#ifdef UNPORTED
	inline unsigned int	coordIndex() const;
	const TmplCoord	&coord() const;
	const TmplCoord	&coord(const TmplCoordSet *cs) const;
#endif  // UNPORTED
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
