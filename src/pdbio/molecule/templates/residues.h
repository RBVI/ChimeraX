#ifndef templates_residues
#define templates_residues

#include "restmpl.h"
#include <vector>
#include <string>


extern const TmplResidue *
	find_template_residue(const std::string &name, bool start, bool end);

#endif  // templates_residues
