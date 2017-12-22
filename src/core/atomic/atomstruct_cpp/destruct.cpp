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

#define ATOMSTRUCT_EXPORT
#define PYINSTANCE_EXPORT
#include "destruct.h"

namespace atomstruct {

ATOMSTRUCT_IMEX void*  DestructionCoordinator::_destruction_batcher = nullptr;
ATOMSTRUCT_IMEX void*  DestructionCoordinator::_destruction_parent = nullptr;
ATOMSTRUCT_IMEX std::set<DestructionObserver*>  DestructionCoordinator::_observers;
ATOMSTRUCT_IMEX std::set<void*>  DestructionCoordinator::_destroyed;
ATOMSTRUCT_IMEX int DestructionCoordinator::_num_notifications_off = 0;

}  // namespace atomstruct
