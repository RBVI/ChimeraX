// vi: set expandtab ts=4 sw=4:
#include "restmpl.h"
#include "Molecule.h"

namespace tmpl {

int
Atom::new_coord(const Coord &c) const
{
    unsigned int    index = COORD_UNASSIGNED;

    const Molecule::CoordSets &css = molecule()->coord_sets();
    for (Molecule::CoordSets::const_iterator csi = css.begin();
                        csi != css.end(); ++csi) {
        CoordSet *cs = *csi;
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
Atom::set_coord(const Coord &c)
{
    CoordSet *cs;
    if ((cs = molecule()->active_coord_set()) == NULL) {
        int csid = 0;
        if ((cs = molecule()->find_coord_set(csid)) == NULL)
            cs = molecule()->new_coord_set(csid);
        molecule()->set_active_coord_set(cs);
    }
    set_coord(c, cs);
}

void
Atom::set_coord(const Coord &c, CoordSet *cs)
{
    if (molecule()->active_coord_set() == NULL)
        molecule()->set_active_coord_set(cs);
    if (_index == COORD_UNASSIGNED)
        _index = new_coord(c);
    else if (_index >= cs->coords().size()) {
        if (_index > cs->coords().size()) {
            CoordSet *prev_cs = molecule()->find_coord_set(cs->id()-1);
            while (_index > cs->coords().size())
                if (prev_cs == NULL)
                    cs->add_coord(Coord());
                else
                    cs->add_coord(*(prev_cs->find_coord(cs->coords().size())));
        }
        cs->add_coord(c);
    } else {
        Coord *cp = cs->find_coord(_index);
        *cp = c;
    }
}

Atom::Atom(Molecule *_owner_, const AtomName& n, const Element& e):
    _element(&e), _index(COORD_UNASSIGNED), _molecule(_owner_), _name(n),
    _residue(0)

{
}

Atom::~Atom()
{
}

}  // namespace tmpl
