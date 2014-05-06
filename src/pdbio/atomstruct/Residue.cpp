// vim: set expandtab ts=4 sw=4:
#include "Residue.h"
#include "Atom.h"
#include <utility>  // for pair
#include <sstream>
#include <set>

Residue::Residue(AtomicStructure *as, std::string &name, std::string &chain,
    int pos, char insert): _structure(as), _name(name), _position(pos),
    _chain_id(chain), _insertion_code(insert), _is_helix(false),
    _is_sheet(false), _is_het(false), _ss_id(-1), _alt_loc(' ')
{
}

void
Residue::add_atom(Atom *a)
{
    a->_residue = this;
    _atoms.push_back(a);
}

Residue::AtomsMap
Residue::atoms_map() const
{
    AtomsMap map;
    for (Atoms::const_iterator ai=_atoms.begin(); ai != _atoms.end(); ++ai) {
        Atom *a = *ai;
        map.insert(AtomsMap::value_type(a->name(), a));
    }
    return map;
}

int
Residue::count_atom(const std::string &name) const
{
    int count = 0;
    for (Atoms::const_iterator ai=_atoms.begin(); ai != _atoms.end(); ++ai) {
        Atom *a = *ai;
        if (a->name() == name)
            ++count;
    }
    return count;
}

int
Residue::count_atom(const char *name) const
{
    int count = 0;
    for (Atoms::const_iterator ai=_atoms.begin(); ai != _atoms.end(); ++ai) {
        Atom *a = *ai;
        if (a->name() == name)
            ++count;
    }
    return count;
}

Atom *
Residue::find_atom(const std::string &name) const
{
    for (Atoms::const_iterator ai=_atoms.begin(); ai != _atoms.end(); ++ai) {
        Atom *a = *ai;
        if (a->name() == name)
            return a;
    }
    return NULL;
}

Atom *
Residue::find_atom(const char *name) const
{
    
    for (Atoms::const_iterator ai=_atoms.begin(); ai != _atoms.end(); ++ai) {
        Atom *a = *ai;
        if (a->name() == name)
            return a;
    }
    return NULL;
}

void
Residue::set_alt_loc(char alt_loc)
{
    if (alt_loc == _alt_loc || alt_loc == ' ') return;
    std::set<Residue *> nb_res;
    bool have_alt_loc = false;
    for (Atoms::const_iterator ai=_atoms.begin(); ai != _atoms.end(); ++ai) {
        Atom *a = *ai;
        if (a->has_alt_loc(alt_loc)) {
            a->set_alt_loc(alt_loc, false, true);
            have_alt_loc = true;
            const Atom::BondsMap &bm = a->bonds_map();
            for (auto bi = bm.begin(); bi != bm.end(); ++bi) {
                Atom *nb = (*bi).first;
                if (nb->residue() != this && nb->has_alt_loc(alt_loc))
                    nb_res.insert(nb->residue());
            }
        }
    }
    if (!have_alt_loc) {
        std::stringstream msg;
        msg << "set_alt_loc(): residue " << str()
            << " does not have an alt loc '" << alt_loc << "'";
        throw std::invalid_argument(msg.str().c_str());
    }
    _alt_loc = alt_loc;
    for (auto nri = nb_res.begin(); nri != nb_res.end(); ++nri) {
        (*nri)->set_alt_loc(alt_loc);
    }
}

std::string
Residue::str() const
{
    std::stringstream pos_string;
    std::string ret = _name;
    ret += " ";
    pos_string << _position;
    ret += pos_string.str();
    if (_insertion_code != ' ') {
        ret += ".";
        ret += _insertion_code;
    }
    return ret;
}
