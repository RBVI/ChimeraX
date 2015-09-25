// vi: set expandtab ts=4 sw=4:
#include "restmpl.h"

#include <algorithm> // for std::find

namespace tmpl {

void
Molecule::set_active_coord_set(CoordSet *cs)
{
    CoordSet *new_active;
    if (cs == NULL) {
        if (_coord_sets.empty())
            return;
        new_active = _coord_sets.front();
    } else {
        CoordSets::iterator csi = std::find(_coord_sets.begin(), _coord_sets.end(), cs);
        if (csi == _coord_sets.end()) {
            throw std::out_of_range("Request active template coord set not in list");
        } else {
            new_active = cs;
        }
    }
    _active_cs = new_active;
}

Atom *
Molecule::new_atom(const AtomName& n, const element::Element& e)
{
    Atom *_inst_ = new Atom(this, n, e);
    _atoms.insert(_inst_);
    return _inst_;
}

Bond *
Molecule::new_bond(Atom *a0, Atom *a1)
{
    Bond *_inst_ = new Bond(this, a0, a1);
    _bonds.insert(_inst_);
    return _inst_;
}

CoordSet *
Molecule::new_coord_set(int k)
{
    CoordSet *_inst_ = new CoordSet(this, k);
    if ((int)_coord_sets.size() <= _inst_->id())
        _coord_sets.resize(_inst_->id() + 1, NULL);
    _coord_sets[_inst_->id()] = _inst_;
    return _inst_;
}

CoordSet *
Molecule::find_coord_set(int index) const
{
    for (CoordSets::const_iterator csi = _coord_sets.begin(); csi != _coord_sets.end();
    ++csi) {
        if ((*csi)->id() == index)
            return *csi;
    }
    return NULL;
}

Residue *
Molecule::new_residue(const char *t)
{
    Residue *_inst_ = new Residue(this, t);
    _residues[_inst_->name()] = _inst_;
    return _inst_;
}

Residue *
Molecule::find_residue(const ResName& index) const
{
    Residues::const_iterator i = _residues.find(index);
    if (i == _residues.end())
        return NULL;
    return i->second;
}

Molecule::Molecule(): _active_cs(NULL)
{
}

Molecule::~Molecule()
{
    for (Residues::iterator j = _residues.begin(); j != _residues.end(); ++j) {
        delete (*j).second;
    }
    for (CoordSets::iterator j = _coord_sets.begin(); j != _coord_sets.end(); ++j) {
        delete (*j);
    }
    for (Bonds::iterator j = _bonds.begin(); j != _bonds.end(); ++j) {
        delete (*j);
    }
    for (Atoms::iterator j = _atoms.begin(); j != _atoms.end(); ++j) {
        delete (*j);
    }
}

}  // namespace tmpl
