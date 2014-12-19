// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_connect
#define atomstruct_connect

#include <string>
#include <vector>
#include <set>
#include "imex.h"
#include "AtomicStructure.h"
#include "Residue.h"
#include "Atom.h"
#include "MolResId.h"
#include "tmpl/Residue.h"

namespace atomstruct {

ATOMSTRUCT_IMEX bool  standard_residue(const std::string& name);
ATOMSTRUCT_IMEX void  add_standard_residue(const std::string& name);
ATOMSTRUCT_IMEX void  remove_standard_residue(const std::string& name);

ATOMSTRUCT_IMEX void  connect_structure(AtomicStructure* as,
        std::vector<Residue *>* chain_starters,
        std::vector<Residue *>* chain_enders,
        std::set<Atom *>* conect_atoms, std::set<MolResId>* mod_res);
ATOMSTRUCT_IMEX void connect_residue_by_distance(Residue* r,
        std::set<Atom *>* conect_atoms = NULL);

}  // namespace atomstruct

#endif  // atomstruct_connect
