// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This particular file is part of the ChimeraX library. You can also
 * redistribute and/or modify it under the terms of the GNU Lesser General
 * Public License version 2.1 as published by the Free Software Foundation.
 * For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
 * LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
 * VERSION 2.1
 *
 * This notice must be embedded in or attached to all copies, including partial
 * copies, of the software or any revisions or derivations thereof.
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

}  // namespace tmpl

#endif  // templates_TAexcept
