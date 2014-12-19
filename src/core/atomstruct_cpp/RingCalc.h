// vim: set expandtab ts=4 sw=4:
#ifndef atomstruct_RingCalc
#define atomstruct_RingCalc

#include "AtomicStructure.h"

namespace atomstruct {

void calculate_rings(const AtomicStructure& as, bool cross_residues,
    unsigned int all_size_threshold, AtomicStructure::Rings& calculated_rings,
    std::set<const Residue *>* ignore);

} // namespace atomstruct

#endif  // atomstruct_RingCalc
