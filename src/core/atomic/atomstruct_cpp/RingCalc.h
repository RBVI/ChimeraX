// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_RingCalc
#define atomstruct_RingCalc

#include "Graph.h"

namespace atomstruct {

void calculate_rings(const Graph& as, bool cross_residues,
    unsigned int all_size_threshold, Graph::Rings& calculated_rings,
    std::set<const Residue *>* ignore);

} // namespace atomstruct

#endif  // atomstruct_RingCalc
