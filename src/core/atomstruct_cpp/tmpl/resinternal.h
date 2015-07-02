// vi: set expandtab ts=4 sw=4:
#ifndef templates_resinternal
#define templates_resinternal

#include <map>
#include <string>

#include "restmpl.h"
#include "../string_types.h"

namespace tmpl {

using atomstruct::ResName;

struct ResInit {
    Residue    *(*start)(Molecule *);
    Residue    *(*middle)(Molecule *);
    Residue    *(*end)(Molecule *);
    ResInit(): start(0), middle(0), end(0) {}
};

typedef std::map<ResName, ResInit> ResInitMap;

}  // namespace tmpl

#endif  // templates_resinternal
