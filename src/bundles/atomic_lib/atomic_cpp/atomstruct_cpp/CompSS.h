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

#ifndef atomstruct_CompSS
#define atomstruct_CompSS

#include <vector>
#include <utility> // std::pair

#include "imex.h"

namespace atomstruct {

class Residue;

class ATOMSTRUCT_IMEX CompSSInfo
{
public:
    std::vector<std::pair<Residue*, Residue*>> strands;
    std::vector<std::set<int>> sheets; // indices into strands
    std::map<std::pair<int, int>, bool> strands_parallel; // indices into strands (false if anti-parallel)
    std::vector<std::pair<std::pair<Residue*, Residue*>, char>> helix_info;
        // helix ends, helix type using same characters as "dssp report true" (G, H, I)
};

}  // namespace atomstruct

#endif  // atomstruct_CompSS
