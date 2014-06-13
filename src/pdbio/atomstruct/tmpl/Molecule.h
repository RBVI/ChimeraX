// vim: set expandtab ts=4 sw=4:
#ifndef templates_TmplMolecule
#define    templates_TmplMolecule

#include <set>
#include <map>
#include <vector>
#include <string>
#include "TAexcept.h"
#include "TmplAtom.h"
#include "TmplBond.h"
#include "TmplCoordSet.h"
#include "TmplResidue.h"
#include "../imex.h"

class ATOMSTRUCT_IMEX TmplMolecule {
public:
        ~TmplMolecule();
    TmplAtom    *new_atom(std::string n, Element e);
    typedef std::set<TmplAtom *> Atoms;
    typedef std::set<TmplBond *> Bonds;
    typedef std::vector<TmplCoordSet *> CoordSets;
    typedef std::map<std::string, TmplResidue *> Residues;
private:
    Atoms    _atoms;
    Bonds    _bonds;
    CoordSets    _coord_sets;
    Residues    _residues;
public:
    TmplBond    *new_bond(TmplAtom *a0, TmplAtom *a1);
    TmplCoordSet    *new_coord_set(int key);
    const CoordSets    &coord_sets() const { return _coord_sets; }
    TmplCoordSet    *find_coord_set(int) const;
    TmplResidue    *new_residue(const char *t);
    TmplResidue    *find_residue(const std::string &) const;
    void        set_active_coord_set(TmplCoordSet *cs);
    TmplCoordSet    *active_coord_set() const { return _active_cs; }
private:
    TmplCoordSet    *_active_cs;
public:
    TmplMolecule();
};

#endif  // templates_TmplMolecule
