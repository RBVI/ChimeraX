// vim: set expandtab ts=4 sw=4:
#include "restmpl.h"
#include "TmplMolecule.h"

int
TmplAtom::new_coord(const TmplCoord &c) const
{
    unsigned int    index = COORD_UNASSIGNED;

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

TmplAtom::TmplAtom(TmplMolecule *_owner_, std::string &n, Element e): _molecule(_owner_), _residue(0), _name(n), _element(e), _index(COORD_UNASSIGNED)

{
}

TmplAtom::~TmplAtom()
{
}

