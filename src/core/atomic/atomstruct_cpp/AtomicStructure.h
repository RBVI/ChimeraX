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

#ifndef atomstruct_AtomicStructure
#define atomstruct_AtomicStructure

#include "Structure.h"

namespace atomstruct {

class ATOMSTRUCT_IMEX AtomicStructure: public Structure {
    friend class Atom; // for IDATM stuff and structure categories
    friend class Bond; // for checking if make_chains() has been run yet, struct categories
    friend class Residue; // for _polymers_computed
    friend class StructureSeq; // for remove_chain()
private:
    void  _compute_atom_types();
    void  _compute_structure_cats() const;
public:
    AtomicStructure(PyObject* logger = nullptr) : Structure(logger) {}

    void  compute_secondary_structure(float energy_cutoff = -0.5, int min_helix_length = 3,
        int min_strand_length = 3, bool report = false);
    AtomicStructure*  copy() const;
    void  make_chains() const;
    std::vector<Chain::Residues>  polymers(bool consider_missing_structure = true,
        bool consider_chain_ids = true) const;
};

}  // namespace atomstruct

#endif  // atomstruct_AtomicStructure
