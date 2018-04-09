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
