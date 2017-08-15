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
#include "Atom.h"
#include "ChangeTracker.h"
#include "PBGroup.h"
#include "PBManager.h"
#include "Pseudobond.h"

namespace atomstruct {

ChangeTracker*
Pseudobond::change_tracker() const { return atoms()[0]->change_tracker(); }

GraphicsChanges*
Pseudobond::graphics_changes() const { return static_cast<GraphicsChanges*>(group()); }

void
Pseudobond::session_restore(int version, int** ints, float** floats) {
    Connection::session_restore(session_base_version(version), ints, floats);
    auto& int_ptr = *ints;
    auto id = int_ptr[0];
    int_ptr += SESSION_NUM_INTS(version);
    auto ses_map = group()->manager()->session_restore_pbs;
    (*ses_map)[id] = this;
}

void
Pseudobond::session_save(int** ints, float** floats) const {
    Connection::session_save(ints, floats);
    auto& int_ptr = *ints;
    // needed to uniquely identify pseudobond upon restore;
    // IDs issued during group->session_save_setup
    auto ses_map = group()->manager()->session_save_pbs;
    int_ptr[0] = (*ses_map)[this];
    int_ptr += SESSION_NUM_INTS();
}

}  // namespace atomstruct
