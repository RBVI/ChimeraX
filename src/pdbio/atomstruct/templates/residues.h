// vim: set expandtab ts=4 sw=4:
#ifndef templates_residues
#define templates_residues

#include "restmpl.h"
#include <vector>
#include <string>
#include "../imex.h"


ATOMSTRUCT_IMEX extern const TmplResidue *
    find_template_residue(const std::string &name, bool start, bool end);

#endif  // templates_residues
