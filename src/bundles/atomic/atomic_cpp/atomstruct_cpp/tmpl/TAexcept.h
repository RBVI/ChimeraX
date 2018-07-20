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

#ifndef templates_TAexcept
#define templates_TAexcept

#include <stdexcept>
#include "../imex.h"

namespace tmpl {
    
// things templateAssign() can throw...
class ATOMSTRUCT_IMEX TA_exception : public std::runtime_error {
public:
    TA_exception(const std::string &msg) : std::runtime_error(msg) {}
};
class ATOMSTRUCT_IMEX TA_TemplateSyntax : public TA_exception {
public:
    TA_TemplateSyntax(const std::string &msg) : TA_exception(msg) {}
};
class ATOMSTRUCT_IMEX TA_NoTemplate : public TA_exception {
public:
    TA_NoTemplate(const std::string &msg) : TA_exception(msg) {}
};

#endif  // templates_TAexcept

}  // namespace tmpl
