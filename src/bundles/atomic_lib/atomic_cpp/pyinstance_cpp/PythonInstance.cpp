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

#define PYINSTANCE_EXPORT
#define PYINSTANCE_MAP_EXPORT
#include "PythonInstance.declare.h"

namespace pyinstance {

// RAII for Python GIL
static PyGILState_STATE gil_state;

AcquireGIL::AcquireGIL() {
    gil_state = PyGILState_Ensure();
}

AcquireGIL::~AcquireGIL() {
    PyGILState_Release(gil_state);
}

PYINSTANCE_IMEX std::map<const void*, PyObject*>  _pyinstance_object_map;

} //  namespace atomstruct
