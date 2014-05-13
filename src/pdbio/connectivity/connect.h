// vim: set expandtab ts=4 sw=4:
#ifndef connectivity_connect
#define connectivity_connect

#include <string>
#include <vector>
#include <set>
#include "imex.h"
#include "MolResId.h"

CONNECTIVITY_IMEX bool  standard_residue(const std::string &name);
CONNECTIVITY_IMEX void  add_standard_residue(const std::string &name);
CONNECTIVITY_IMEX void  remove_standard_residue(const std::string &name);

class AtomicStructure;
class Atom;

CONNECTIVITY_IMEX void  connect_structure(AtomicStructure *as,
        std::vector<Residue *> *chain_starters,
        std::vector<Residue *> *chain_enders,
        std::set<Atom *> *conect_atoms, std::set<MolResId> *mod_res);

#endif  // connectivity_connect
