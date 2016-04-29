// vi: set expandtab ts=4 sw=4:

#define ATOMSTRUCT_EXPORT
#include "Atom.h"
#include "ChangeTracker.h"
#include "PBGroup.h"
#include "PBManager.h"
#include "Pseudobond.h"

namespace atomstruct {

ChangeTracker*
Pseudobond::change_tracker() const { return atoms()[0]->change_tracker(); }

GraphicsContainer*
Pseudobond::graphics_container() const { return static_cast<GraphicsContainer*>(group()); }

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
    auto ses_map = group()->manager()->session_save_pbs;
    int id = ses_map->size();
    (*ses_map)[this] = id;
    int_ptr[0] = id; // needed to uniquely identify pseudobond upon restore
    int_ptr += SESSION_NUM_INTS();
}

}  // namespace atomstruct
