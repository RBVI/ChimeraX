// vi: set expandtab ts=4 sw=4:

#include "Atom.h"
#include "PBGroup.h"
#include "PBManager.h"
#include "Pseudobond.h"

namespace atomstruct {

basegeom::ChangeTracker*
Pseudobond::change_tracker() const { return atoms()[0]->change_tracker(); }

basegeom::GraphicsContainer*
Pseudobond::graphics_container() const { return static_cast<GraphicsContainer*>(group()); }

void
Pseudobond::session_note_atoms(int** ints) const
{
    auto int_ptr = *ints;
    int_ptr[0] = (*(atoms()[0]->structure()->session_save_atoms))[atoms()[0]];
    int_ptr[1] = (*(atoms()[0]->structure()->session_save_atoms))[atoms()[1]];
    int_ptr += 2;
}

void
Pseudobond::session_note_structures(int** ints) const
{
    AtomicStructure* s1 = atoms()[0]->structure();
    AtomicStructure* s2 = atoms()[1]->structure();
    int s1_id, s2_id;
    auto ss_map = group()->manager()->ses_struct_map();
    if (ss_map->find(s1) == ss_map->end()) {
        s1_id = ss_map->size();
        (*ss_map)[s1] = s1_id;
    } else {
        s1_id = (*ss_map)[s1];
    }
    if (ss_map->find(s2) == ss_map->end()) {
        s2_id = ss_map->size();
        (*ss_map)[s2] = s2_id;
    } else {
        s2_id = (*ss_map)[s2];
    }
    auto int_ptr = *ints;
    int_ptr[0] = s1_id;
    int_ptr[1] = s2_id;
    int_ptr += 2;
}

}  // namespace atomstruct
