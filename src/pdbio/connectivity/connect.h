// vim: set expandtab ts=4 sw=4:
#ifndef connectivity_connect
#define connectivity_connect

#include <string>
#include <vector>
#include <set>
#include "imex.h"
#include "atomstruct/AtomicStructure.h"
#include "atomstruct/Residue.h"
#include "atomstruct/Atom.h"
#include "MolResId.h"

namespace connectivity {

CONNECTIVITY_IMEX bool  standard_residue(const std::string &name);
CONNECTIVITY_IMEX void  add_standard_residue(const std::string &name);
CONNECTIVITY_IMEX void  remove_standard_residue(const std::string &name);

CONNECTIVITY_IMEX void  connect_structure(atomstruct::AtomicStructure *as,
        std::vector<atomstruct::Residue *> *chain_starters,
        std::vector<atomstruct::Residue *> *chain_enders,
        std::set<atomstruct::Atom *> *conect_atoms,
        std::set<MolResId> *mod_res);

}  // namespace connectivity

#endif  // connectivity_connect
