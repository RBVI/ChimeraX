// vim: set expandtab ts=4 sw=4:
#ifndef templates_residues
#define templates_residues

#include "restmpl.h"
#include <vector>
#include <string>
#include "../imex.h"

namespace tmpl {

ATOMSTRUCT_IMEX extern const Residue *
    find_template_residue(const std::string &name, bool start, bool end);

}  // namespace tmpl

#endif  // templates_residues
