// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * This software is provided pursuant to the ChimeraX license agreement, which
 * covers academic and commercial uses. For more information, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This file is part of the ChimeraX library. You can also redistribute and/or
 * modify it under the GNU Lesser General Public License version 2.1 as
 * published by the Free Software Foundation. For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * This file is distributed WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
 * must be embedded in or attached to all copies, including partial copies, of
 * the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

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
