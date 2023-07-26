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
#include "Atom.h"
#include "ChangeTracker.h"
#include "PBGroup.h"
#include "PBManager.h"
#include "Pseudobond.h"

#include <pyinstance/PythonInstance.instantiate.h>
template class pyinstance::PythonInstance<atomstruct::Pseudobond>;

namespace atomstruct {

Pseudobond::Pseudobond(Atom* a1, Atom* a2, PBGroup* grp): Connection(a1, a2), _group(grp),
        _shown_when_atoms_hidden(true) {
    _halfbond = false;
    _radius = 0.05;
    change_tracker()->add_created(grp->structure(), this);
    graphics_changes()->set_gc_adddel();
}

ChangeTracker*
Pseudobond::change_tracker() const { return group()->manager()->change_tracker(); }

void
Pseudobond::copy_style(const Pseudobond* exemplar)
{
    set_color(exemplar->color());
    set_display(exemplar->display());
    set_halfbond(exemplar->halfbond());
    set_hide(exemplar->hide());
    set_radius(exemplar->radius());
    set_selected(exemplar->selected());
}

GraphicsChanges*
Pseudobond::graphics_changes() const { return static_cast<GraphicsChanges*>(group()); }

void
Pseudobond::session_restore(int version, int** ints, float** floats) {
    Connection::session_restore(version, ints, floats);
    auto& int_ptr = *ints;
    auto id = int_ptr[0];
    _shown_when_atoms_hidden = version < 9 ? true : int_ptr[1];
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
    int_ptr[1] = _shown_when_atoms_hidden;
    int_ptr += SESSION_NUM_INTS();
}

}  // namespace atomstruct
