// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_AtomicStructure
#define atomstruct_AtomicStructure

#include "Structure.h"

namespace atomstruct {

class ATOMSTRUCT_IMEX AtomicStructure: public Structure {
    friend class Atom; // for IDATM stuff and structure categories
    friend class Bond; // for checking if make_chains() has been run yet, struct categories
    friend class Chain; // for remove_chain()
    friend class Residue; // for _polymers_computed
private:
    void  _compute_atom_types();
    void  _compute_structure_cats() const;
public:
    AtomicStructure(PyObject* logger = nullptr) : Structure(logger) {}

    AtomicStructure*  copy() const;
    void  make_chains() const;
    std::vector<Chain::Residues>  polymers(bool consider_missing_structure = true,
        bool consider_chain_ids = true) const;
};

}  // namespace atomstruct

#endif  // atomstruct_AtomicStructure
