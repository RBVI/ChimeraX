#ifndef templates_resinternal
#define templates_resinternal

#include "restmpl.h"
#include <map>
#include <string>

struct ResInit {
	TmplResidue	*(*start)(TmplMolecule *);
	TmplResidue	*(*middle)(TmplMolecule *);
	TmplResidue	*(*end)(TmplMolecule *);
	ResInit(): start(0), middle(0), end(0) {}
};

typedef std::map<std::string, ResInit> ResInitMap;

#endif  // templates_resinternal
