// vi: set expandtab ts=4 sw=4:
#ifndef templates_residues
#define templates_residues

#include "../imex.h"
#include "restmpl.h"
#include <vector>
#include <string>

#include "../string_types.h"

namespace tmpl {

using atomstruct::ResName;

ATOMSTRUCT_IMEX extern const Residue *
    find_template_residue(const ResName& name, bool start, bool end);

}  // namespace tmpl

#endif  // templates_residues
