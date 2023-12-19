// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This particular file is part of the ChimeraX library. You can also
 * redistribute and/or modify it under the terms of the GNU Lesser General
 * Public License version 2.1 as published by the Free Software Foundation.
 * For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
 * LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
 * VERSION 2.1
 *
 * This notice must be embedded in or attached to all copies, including partial
 * copies, of the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#ifndef atomstruct_AtomicStructure
#define atomstruct_AtomicStructure

#include "Structure.h"

namespace atomstruct {

class ATOMSTRUCT_IMEX AtomicStructure: public Structure {
    friend class Atom; // for IDATM stuff and structure categories
    friend class Bond; // struct categories
    friend class Residue; // for _polymers_computed
    friend class StructureSeq; // for remove_chain()
private:
    void  _compute_atom_types();
    void  _compute_structure_cats() const;
    void  _make_chains() const;
public:
    AtomicStructure(PyObject* logger = nullptr) : Structure(logger) {}

    void  compute_secondary_structure(float energy_cutoff = -0.5, int min_helix_length = 3,
        int min_strand_length = 3, bool = false, CompSSInfo* = nullptr);
    AtomicStructure*  copy() const;
    void  normalize_ss_ids();
    std::vector<std::pair<Chain::Residues,PolymerType>>  polymers(
        PolymerMissingStructure missing_structure_treatment = PMS_ALWAYS_CONNECTS,
        bool consider_chain_ids = true) const;
};

}  // namespace atomstruct

#endif  // atomstruct_AtomicStructure
