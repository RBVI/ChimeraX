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

#ifndef pdbio_connect
#define pdbio_connect

#include <vector>
#include <set>

#include <atomstruct/Atom.h>
#include <atomstruct/MolResId.h>
#include <atomstruct/Residue.h>
#include <atomstruct/string_types.h>
#include <atomstruct/Structure.h>
#include <atomstruct/tmpl/Residue.h>

using namespace atomstruct;

bool  is_standard_residue(const ResName& name);
void  add_standard_residue(const ResName& name);
void  remove_standard_residue(const ResName& name);

void  connect_structure(Structure* as,
        std::vector<Residue *>* chain_starters,
        std::vector<Residue *>* chain_enders,
        std::set<Atom *>* conect_atoms, std::set<MolResId>* mod_res);
void connect_residue_by_distance(Residue* r,
        std::set<Atom *>* conect_atoms = nullptr);
Atom* find_closest(Atom* a, Residue* r, float* ret_dist_sq,
        bool nonSaturated=false);
void  find_nearest_pair(Residue* from, Residue* to,
        Atom** ret_from_atom, Atom** ret_to_atom, float* ret_dist_sq = nullptr);
void  find_and_add_metal_coordination_bonds(Structure* as);
void  find_missing_structure_bonds(Structure* as);

#endif  // pdbio_connect
