// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * This software is provided pursuant to the ChimeraX license agreement, which
 * covers academic and commercial uses. For more information, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This file is part of the ChimeraX library. You can also redistribute and/or
 * modify it under the GNU Lesser General Public License version 2.1 as
 * published by the Free Software Foundation. For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * This file is distributed WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
 * must be embedded in or attached to all copies, including partial copies, of
 * the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#ifndef pdbio_connect
#define pdbio_connect

#include "imex.h"

#include <vector>
#include <set>

#include <atomstruct/Atom.h>
#include <atomstruct/MolResId.h>
#include <atomstruct/Residue.h>
#include <atomstruct/string_types.h>
#include <atomstruct/Structure.h>
#include <atomstruct/tmpl/Residue.h>

using namespace atomstruct;

namespace pdb_connect {

PDB_CONNECT_IMEX void connect_structure(Structure* as,
        std::vector<Residue *>* chain_starters,
        std::vector<Residue *>* chain_enders,
        std::set<Atom *>* conect_atoms, std::set<MolResId>* mod_res,
        std::set<ResName>& polymeric_res_names, std::set<Residue*>& het_res);
PDB_CONNECT_IMEX void connect_residue_by_distance(Residue* r,
        std::set<Atom *>* conect_atoms = nullptr);
PDB_CONNECT_IMEX Atom* find_closest(Atom* a, Residue* r, float* ret_dist_sq,
        bool nonSaturated=false);
PDB_CONNECT_IMEX void find_nearest_pair(Residue* from, Residue* to,
        Atom** ret_from_atom, Atom** ret_to_atom, float* ret_dist_sq = nullptr);
PDB_CONNECT_IMEX void find_and_add_metal_coordination_bonds(Structure* as);
PDB_CONNECT_IMEX void find_missing_structure_bonds(Structure* as);

} // namespace pdb_connect

#endif  // pdbio_connect
