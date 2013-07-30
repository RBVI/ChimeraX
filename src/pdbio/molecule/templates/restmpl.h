#ifndef templates_restmpl
#define	templates_restmpl

#include "TmplAtom.h"
#include "TmplBond.h"
#include "TmplCoord.h"
#include "TmplCoordSet.h"
#include "TmplMolecule.h"
#include "TmplResidue.h"

#ifdef UNPORTED
namespace molecule {

inline TmplAtom::Bonds
TmplAtom::bonds() const
{
	std::vector<TmplBond *> result;
	result.reserve(Bonds_.size());
	for (std::map<TmplAtom*, TmplBond *>::const_iterator i = Bonds_.begin(); i != Bonds_.end(); ++i)
		result.push_back(i->second);
	return result;
}
inline const TmplAtom::BondsMap &
TmplAtom::bondsMap() const
{
	return Bonds_;
}
inline TmplAtom::BondKeys
TmplAtom::neighbors() const
{
	std::vector<TmplAtom*> result;
	result.reserve(Bonds_.size());
	for (std::map<TmplAtom*, TmplBond *>::const_iterator i = Bonds_.begin(); i != Bonds_.end(); ++i)
		result.push_back(i->first);
	return result;
}
inline TmplBond *
TmplAtom::connectsTo(TmplAtom *a) const
{
	BondsMap::const_iterator found = bondsMap().find(a);
	if (found == bondsMap().end())
		return NULL;
	else
		return found->second;
}
inline Symbol
TmplAtom::name() const
{
	return name_;
}
inline void
TmplAtom::setName(Symbol s)
{
	name_ = s;
}
inline Element
TmplAtom::element() const
{
	return element_;
}
inline unsigned int
TmplAtom::coordIndex() const
{
	return index_;
}
inline const TmplBond::Atoms &
TmplBond::atoms() const
{
	return Atoms_;
}
inline Real
TmplBond::length() const
{
	return distance(findAtom(1)->coord(), findAtom(0)->coord());
}
inline Real
TmplBond::sqlength() const
{
	return sqdistance(findAtom(1)->coord(), findAtom(0)->coord());
}
inline const TmplCoordSet::Coords &
TmplCoordSet::coords() const
{
	return Coords_;
}
inline int
TmplCoordSet::id() const
{
	return csid;
}
inline const TmplMolecule::Atoms &
TmplMolecule::atoms() const
{
	return Atoms_;
}
inline const TmplMolecule::Bonds &
TmplMolecule::bonds() const
{
	return Bonds_;
}
inline const TmplMolecule::CoordSets &
TmplMolecule::coordSets() const
{
	return CoordSets_;
}
inline TmplMolecule::Residues
TmplMolecule::residues() const
{
	std::vector<TmplResidue *> result;
	result.reserve(Residues_.size());
	for (std::map<Symbol, TmplResidue *>::const_iterator i = Residues_.begin(); i != Residues_.end(); ++i)
		result.push_back(i->second);
	return result;
}
inline const TmplMolecule::ResiduesMap &
TmplMolecule::residuesMap() const
{
	return Residues_;
}
inline TmplMolecule::ResidueKeys
TmplMolecule::residueNames() const
{
	std::vector<Symbol> result;
	result.reserve(Residues_.size());
	for (std::map<Symbol, TmplResidue *>::const_iterator i = Residues_.begin(); i != Residues_.end(); ++i)
		result.push_back(i->first);
	return result;
}
inline TmplCoordSet *
TmplMolecule::activeCoordSet() const
{
	return activeCS;
}
inline TmplResidue::Atoms
TmplResidue::atoms() const
{
	std::vector<TmplAtom *> result;
	result.reserve(Atoms_.size());
	for (std::map<Symbol, TmplAtom *>::const_iterator i = Atoms_.begin(); i != Atoms_.end(); ++i)
		result.push_back(i->second);
	return result;
}
inline const TmplResidue::AtomsMap &
TmplResidue::atomsMap() const
{
	return Atoms_;
}
inline TmplResidue::AtomKeys
TmplResidue::atomNames() const
{
	std::vector<Symbol> result;
	result.reserve(Atoms_.size());
	for (std::map<Symbol, TmplAtom *>::const_iterator i = Atoms_.begin(); i != Atoms_.end(); ++i)
		result.push_back(i->first);
	return result;
}
inline Symbol
TmplResidue::type() const
{
	return type_;
}
inline void
TmplResidue::setType(Symbol t)
{
	type_ = t;
}
inline const MolResId &
TmplResidue::id() const
{
	return rid;
}
inline void
TmplResidue::setId(MolResId &mid)
{
	rid = mid;
}
inline bool
TmplResidue::operator==(const TmplResidue &r) const
{
	// only equal if they are the same residue
	return this == &r;
}
inline bool
TmplResidue::operator<(const TmplResidue &r) const
{
	return rid < r.rid;
}

} // namespace molecule
#endif  // UNPORTED

#endif  // templates_restmpl
