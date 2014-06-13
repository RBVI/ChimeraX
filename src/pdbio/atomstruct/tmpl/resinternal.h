// vim: set expandtab ts=4 sw=4:
#ifndef templates_resinternal
#define templates_resinternal

#include "restmpl.h"
#include <map>
#include <string>

namespace tmpl {

struct ResInit {
    Residue    *(*start)(Molecule *);
    Residue    *(*middle)(Molecule *);
    Residue    *(*end)(Molecule *);
    ResInit(): start(0), middle(0), end(0) {}
};

typedef std::map<std::string, ResInit> ResInitMap;

}  // namespace tmpl

#endif  // templates_resinternal
