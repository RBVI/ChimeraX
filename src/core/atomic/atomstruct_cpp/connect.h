// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2016 Regents of the University of California.
 * All rights reserved.  This software provided pursuant to a
 * license agreement containing restrictions on its disclosure,
 * duplication and use.  For details see:
 * http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
 * This notice must be embedded in or attached to all copies,
 * including partial copies, of the software or any revisions
 * or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#ifndef atomstruct_connect
#define atomstruct_connect

#include <string>
#include <vector>
#include <set>

#include "Atom.h"
#include "Structure.h"
#include "imex.h"
#include "MolResId.h"
#include "Residue.h"
#include "tmpl/Residue.h"

namespace atomstruct {

ATOMSTRUCT_IMEX bool  standard_residue(const std::string& name);
ATOMSTRUCT_IMEX void  add_standard_residue(const std::string& name);
ATOMSTRUCT_IMEX void  remove_standard_residue(const std::string& name);

ATOMSTRUCT_IMEX void  connect_structure(Structure* as,
        std::vector<Residue *>* chain_starters,
        std::vector<Residue *>* chain_enders,
        std::set<Atom *>* conect_atoms, std::set<MolResId>* mod_res);
ATOMSTRUCT_IMEX void connect_residue_by_distance(Residue* r,
        std::set<Atom *>* conect_atoms = nullptr);
ATOMSTRUCT_IMEX Atom* find_closest(Atom* a, Residue* r, float* ret_dist_sq,
        bool nonSaturated=false);
ATOMSTRUCT_IMEX void  find_nearest_pair(Residue* from, Residue* to,
        Atom** ret_from_atom, Atom** ret_to_atom, float* ret_dist_sq = nullptr);
ATOMSTRUCT_IMEX void  find_and_add_metal_coordination_bonds(Structure* as);
ATOMSTRUCT_IMEX void  find_missing_structure_bonds(Structure* as);

}  // namespace atomstruct

#endif  // atomstruct_connect
