// vi: set expandtab ts=4 sw=4:

#include "Atom.h"
#include "AtomicStructure.h"
#include "PBGroup.h"
#include "Pseudobond.h"

#include <basegeom/destruct.h>

#include <Python.h>
#include <pythonarray.h>

namespace atomstruct {

void
StructurePBGroupBase::_check_structure(Atom* a1, Atom* a2)
{
    if (_structure != nullptr && (a1->structure() != _structure || a2->structure() != _structure))
        throw std::invalid_argument("Pseudobond endpoints not in "
            " atomic structure associated with group");
}

static void
_check_destroyed_atoms(Group::Pseudobonds& pbonds, const std::set<void*>& destroyed,
    GraphicsContainer* gc)
{
    Group::Pseudobonds remaining;
    for (auto pb: pbonds) {
        auto& pb_atoms = pb->atoms();
        if (destroyed.find(static_cast<void*>(pb_atoms[0])) != destroyed.end()
        || destroyed.find(static_cast<void*>(pb_atoms[1])) != destroyed.end()) {
            delete pb;
        } else {
            remaining.insert(pb);
        }
    }
    if (remaining.size() == 0) {
        pbonds.clear();
        gc->set_gc_shape();
    } else if (remaining.size() != pbonds.size()) {
        pbonds.swap(remaining);
        gc->set_gc_shape();
    }
}

void
CS_PBGroup::check_destroyed_atoms(const std::set<void*>& destroyed)
{
    auto db = basegeom::DestructionBatcher(this);
    for (auto& cs_pbs: _pbonds)
        _check_destroyed_atoms(cs_pbs.second, destroyed,
            static_cast<GraphicsContainer*>(this));
}

void
StructurePBGroup::check_destroyed_atoms(const std::set<void*>& destroyed)
{
    auto db = basegeom::DestructionBatcher(this);
    _check_destroyed_atoms(_pbonds, destroyed,
        static_cast<GraphicsContainer*>(this));
}

void
CS_PBGroup::clear()
{
    for (auto cat_set : _pbonds)
        for (auto pb: cat_set.second)
            delete pb;
    _pbonds.clear();
}

void
StructurePBGroup::clear()
{
    for (auto pb : _pbonds)
        delete pb;
    _pbonds.clear();
}

CS_PBGroup::~CS_PBGroup()
{
    _destruction_relevant = false;
    auto du = basegeom::DestructionUser(this);
    for (auto name_pbs: _pbonds) {
        for (auto pb: name_pbs.second)
            delete pb;
    }
}

void
Group::dtor_code()
{
    _destruction_relevant = false;
    auto du = basegeom::DestructionUser(this);
    for (auto pb: pseudobonds())
        delete pb;
}

Pseudobond*
CS_PBGroup::new_pseudobond(Atom* a1, Atom* a2)
{
    return new_pseudobond(a1, a2, structure()->active_coord_set());
}

Pseudobond*
CS_PBGroup::new_pseudobond(Atom* a1, Atom* a2, CoordSet* cs)
{
    _check_structure(a1, a2);
    Pseudobond* pb = new Pseudobond(a1, a2, this);
    pb->set_color(get_default_color());
    pb->set_halfbond(get_default_halfbond());
    auto pbi = _pbonds.find(cs);
    if (pbi == _pbonds.end()) {
        _pbonds[cs].insert(pb);
    } else {
        (*pbi).second.insert(pb);
    }
    return pb;
}

Pseudobond*
StructurePBGroup::new_pseudobond(Atom* a1, Atom* a2)
{
    _check_structure(a1, a2);
    Pseudobond* pb = new Pseudobond(a1, a2, this);
    pb->finish_construction();
    pb->set_color(get_default_color());
    pb->set_halfbond(get_default_halfbond());
    _pbonds.insert(pb); return pb;
}

const Group::Pseudobonds&
CS_PBGroup::pseudobonds() const
{
    return pseudobonds(_structure->active_coord_set());
}

void
StructurePBGroupBase::session_note_pb_ctor_info(Pseudobond* pb, int** ints) const
{
    auto& int_ptr = *ints;
    for (auto a: pb->atoms()) {
        auto s = a->structure();
        if (s == nullptr) {
            // note structure info
            auto ss_map = manager()->ses_struct_map();
            int s_id;
            if (ss_map->find(s) == ss_map->end()) {
                s_id = ss_map->size();
                (*ss_map)[s] = s_id;
            } else {
                s_id = (*ss_map)[s];
            }
            int_ptr[0] = s_id;
            int_ptr++;
        }
        int_ptr[0] = (*(s->session_save_atoms))[a];
        int_ptr++;
    }
}

int
CS_PBGroup::session_num_floats() const {
    int num_floats = SESSION_NUM_FLOATS + StructurePBGroupBase::session_num_floats();
    for (auto crdset_pbs: _pbonds) {
        num_floats += crdset_pbs.second.size() * Pseudobond::session_num_floats();
    }
    return num_floats;
}

int
StructurePBGroup::session_num_floats() const {
    return SESSION_NUM_FLOATS + StructurePBGroupBase::session_num_floats()
        + pseudobonds().size() * Pseudobond::session_num_floats();
}

int
CS_PBGroup::session_num_ints() const {
    int num_ints = SESSION_NUM_INTS + StructurePBGroupBase::session_num_ints()
        + pseudobonds().size(); // that last is for references to coord sets
    for (auto crdset_pbs: _pbonds) {
        // the +2 in the next line is for the atom IDs
        num_ints += crdset_pbs.second.size() * (Pseudobond::session_num_ints() + 2);
    }
    return num_ints;
}

int
StructurePBGroup::session_num_ints() const {
    int num_ints = SESSION_NUM_INTS + StructurePBGroupBase::session_num_ints()
        + pseudobonds().size() * (Pseudobond::session_num_ints() + 2); // +2 for the atom IDs
    if (structure() == nullptr) // will need to remember the structure IDs too
        num_ints += pseudobonds().size() * 2;
    return num_ints;
}

void
Group::session_save(int** ints, float** floats) const
{
    _default_color.session_save(ints, floats);
    auto& int_ptr = *ints;
    int_ptr[0] = _default_halfbond;
    int_ptr += SESSION_NUM_INTS;
}

void
CS_PBGroup::session_save(int** ints, float** floats) const
{
    StructurePBGroupBase::session_save(ints, floats);
    for (auto cs_pbs: _pbonds) {
        auto cs = cs_pbs.first;
        auto& pbs = cs_pbs.second;
        auto& int_ptr = *ints;
        int_ptr[0] = (*structure()->session_save_crdsets)[cs];
        int_ptr += 1;
        for (auto pb: pbs) {
            session_note_pb_ctor_info(pb, ints);
            pb->session_save(ints, floats);
        }
    }
}

void
StructurePBGroup::session_save(int** ints, float** floats) const
{
    StructurePBGroupBase::session_save(ints, floats);
    for (auto pb: pseudobonds()) {
        session_note_pb_ctor_info(pb, ints);
        pb->session_save(ints, floats);
    }
}

}  // namespace atomstruct
