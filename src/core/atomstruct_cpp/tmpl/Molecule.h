// vi: set expandtab ts=4 sw=4:
#ifndef templates_Molecule
#define    templates_Molecule

#include <set>
#include <map>
#include <vector>
#include <string>
#include "TAexcept.h"
#include "Atom.h"
#include "Bond.h"
#include "CoordSet.h"
#include "Residue.h"
#include "../imex.h"

namespace tmpl {

class ATOMSTRUCT_IMEX Molecule {
public:
        ~Molecule();
    Atom    *new_atom(const AtomName& n, atomstruct::Element e);
    typedef std::set<Atom *> Atoms;
    typedef std::set<Bond *> Bonds;
    typedef std::vector<CoordSet *> CoordSets;
    typedef std::map<std::string, Residue *> Residues;
private:
    Atoms    _atoms;
    Bonds    _bonds;
    CoordSets    _coord_sets;
    Residues    _residues;
public:
    Bond    *new_bond(Atom *a0, Atom *a1);
    CoordSet    *new_coord_set(int key);
    const CoordSets    &coord_sets() const { return _coord_sets; }
    CoordSet    *find_coord_set(int) const;
    Residue    *new_residue(const char *t);
    Residue    *find_residue(const std::string &) const;
    void        set_active_coord_set(CoordSet *cs);
    CoordSet    *active_coord_set() const { return _active_cs; }
    const Residues &residues_map() { return _residues; }
private:
    CoordSet    *_active_cs;
public:
    Molecule();
};

}  // namespace tmpl

#endif  // templates_Molecule
