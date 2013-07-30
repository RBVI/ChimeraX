#include "restmpl.h"
#include "TmplMolecule.h"

#ifdef UNPORTED
void
TmplAtom::setElement(Element e)
{
	element_ = e;
}
#endif  // UNPORTED


int
TmplAtom::new_coord(const TmplCoord &c) const
{
	unsigned int	index = COORD_UNASSIGNED;

	const TmplMolecule::CoordSets &css = molecule()->coord_sets();
	for (TmplMolecule::CoordSets::const_iterator csi = css.begin();
						csi != css.end(); ++csi) {
		TmplCoordSet *cs = *csi;
		if (index == COORD_UNASSIGNED) {
			index = cs->coords().size();
			cs->add_coord(c);
		}
		else while (index >= cs->coords().size())
			cs->add_coord(c);
	}
	return index;
}

#ifdef UNPORTED
const TmplCoord &
TmplAtom::coord() const
{

	if (index_ == COORD_UNASSIGNED)
		throw std::logic_error("coordinate value not set yet");
	TmplCoordSet *cs;
	if ((cs = molecule()->activeCoordSet()) == NULL)
		throw std::logic_error("no active coordinate set");
	return *cs->findCoord(index_);
}

const TmplCoord &
TmplAtom::coord(const TmplCoordSet *cs) const
{
	if (index_ == COORD_UNASSIGNED)
		throw std::logic_error("coordinate value not set yet");
	// since genlib doesn't current generate a const version of
	// findCoord, cast away const
	return *((TmplCoordSet *)cs)->findCoord(index_);
}
#endif  // UNPORTED

void
TmplAtom::set_coord(const TmplCoord &c)
{
	TmplCoordSet *cs;
	if ((cs = molecule()->active_coord_set()) == NULL) {
		int csid = 0;
		if ((cs = molecule()->find_coord_set(csid)) == NULL)
			cs = molecule()->new_coord_set(csid);
		molecule()->set_active_coord_set(cs);
	}
	set_coord(c, cs);
}

void
TmplAtom::set_coord(const TmplCoord &c, TmplCoordSet *cs)
{
	if (molecule()->active_coord_set() == NULL)
		molecule()->set_active_coord_set(cs);
	if (_index == COORD_UNASSIGNED)
		_index = new_coord(c);
	else if (_index >= cs->coords().size()) {
		if (_index > cs->coords().size()) {
			TmplCoordSet *prev_cs = molecule()->find_coord_set(cs->id()-1);
			while (_index > cs->coords().size())
				if (prev_cs == NULL)
					cs->add_coord(TmplCoord());
				else
					cs->add_coord(*(prev_cs->find_coord(cs->coords().size())));
		}
		cs->add_coord(c);
	} else {
		TmplCoord *cp = cs->find_coord(_index);
		*cp = c;
	}
}

void
TmplAtom::add_bond(TmplBond *b)
{
	_bonds[b->other_atom(this)] = b;
}
#ifdef UNPORTED
void
TmplAtom::removeBond(TmplBond *element)
{
	Bonds_.erase(element->otherAtom(this));
}
TmplBond *
TmplAtom::findBond(TmplAtom* index) const
{
	BondsMap::const_iterator i = Bonds_.find(index);
	if (i == Bonds_.end())
		return NULL;
	return i->second;
}
TmplMolecule *
TmplAtom::molecule() const
{
	return Molecule_;
}
TmplResidue *
TmplAtom::residue() const
{
	return Residue_;
}
#endif  // UNPORTED
TmplAtom::TmplAtom(TmplMolecule *_owner_, std::string &n, Element e): _molecule(_owner_), _residue(0), _name(n), _element(e), _index(COORD_UNASSIGNED)

{
}

TmplAtom::~TmplAtom()
{
}

