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

#include <ctype.h>
#include "cmp_nocase.h"

namespace chutil {

int
cmp_nocase(const std::string &s, const std::string &s2)
{
    std::string::const_iterator p = s.begin();
    std::string::const_iterator p2 = s2.begin();

    while (p != s.end() && p2 != s2.end()) {
        char c = toupper(*p);
        char c2 = toupper(*p2);
        if (c != c2)
            return c < c2 ? -1 : 1;
        ++p;
        ++p2;
    }
    return s2.size() - s.size();
}

}  // namespace chutil
