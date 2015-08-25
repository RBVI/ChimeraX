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
        std::set<Atom *>* conect_atoms = nullptr);
ATOMSTRUCT_IMEX Atom* find_closest(Atom* a, Residue* r, float* ret_dist_sq,
        bool nonSaturated=false);
ATOMSTRUCT_IMEX void  find_nearest_pair(Residue* from, Residue* to,
        Atom** ret_from_atom, Atom** ret_to_atom, float* ret_dist_sq = nullptr);
ATOMSTRUCT_IMEX void  find_and_add_metal_coordination_bonds(AtomicStructure* as);

}  // namespace atomstruct

#endif  // atomstruct_connect
