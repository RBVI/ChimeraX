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
#include "destruct.h"
#include "Structure.h"
#include "PBGroup.h"
#include "Pseudobond.h"

#include <Python.h>
#include <arrays/pythonarray.h>

namespace atomstruct {

void
StructurePBGroupBase::_check_structure(Atom* a1, Atom* a2)
{
    if (_structure != nullptr && (a1->structure() != _structure || a2->structure() != _structure))
        throw std::invalid_argument("Pseudobond endpoints not in "
            " atomic structure associated with group");
}

void
PBGroup::_check_destroyed_atoms(PBGroup::Pseudobonds& pbonds, const std::set<void*>& destroyed)
{
    PBGroup::Pseudobonds remaining;
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
        set_gc_shape();
    } else if (remaining.size() != pbonds.size()) {
        pbonds.swap(remaining);
        set_gc_shape();
    }
}

void
CS_PBGroup::check_destroyed_atoms(const std::set<void*>& destroyed)
{
    auto db = DestructionBatcher(this);
    for (auto& cs_pbs: _pbonds)
        _check_destroyed_atoms(cs_pbs.second, destroyed);
}

void
StructurePBGroup::check_destroyed_atoms(const std::set<void*>& destroyed)
{
    auto db = DestructionBatcher(this);
    _check_destroyed_atoms(_pbonds, destroyed);
}

void
CS_PBGroup::clear()
{
    auto db = DestructionBatcher(this);
    for (auto cat_set : _pbonds)
        for (auto pb: cat_set.second)
            delete pb;
    _pbonds.clear();
}

void
StructurePBGroup::clear()
{
    auto db = DestructionBatcher(this);
    for (auto pb : _pbonds)
        delete pb;
    _pbonds.clear();
}

void
CS_PBGroup::change_cs(const CoordSet*)
{
    if (structure()->active_coord_set_change_notify())
        set_gc_display();
}

CS_PBGroup::~CS_PBGroup()
{
    _destruction_relevant = false;
    auto du = DestructionUser(this);
    for (auto name_pbs: _pbonds) {
        for (auto pb: name_pbs.second)
            delete pb;
    }
}

void
PBGroup::delete_pbs_check(const std::set<Pseudobond*>& pbs) const
{
    for (auto pb: pbs)
        if (pb->group() != this)
            throw std::invalid_argument("delete_pseudobonds called for Pseudobond not in PBGroup");
}

void
CS_PBGroup::delete_pseudobond(Pseudobond* pb)
{
    if (pb->group() != this)
        throw std::invalid_argument("delete_pseudobond called for Pseudobond not in PBGroup");

    auto db = DestructionBatcher(this);
    auto cs = static_cast<CS_Pseudobond*>(pb)->coord_set();
    _pbonds[cs].erase(pb);
    delete pb;
    set_gc_shape();
}

void
StructurePBGroup::delete_pseudobond(Pseudobond* pb)
{
    if (pb->group() != this)
        throw std::invalid_argument("delete_pseudobond called for Pseudobond not in PBGroup");

    auto db = DestructionBatcher(this);
    _pbonds.erase(pb);
    delete pb;
    set_gc_shape();
}

void
CS_PBGroup::delete_pseudobonds(const std::set<Pseudobond*>& pbs)
{
    delete_pbs_check(pbs);
    auto db = DestructionBatcher(this);
    for (auto pb: pbs) {
        _pbonds[static_cast<CS_Pseudobond*>(pb)->coord_set()].erase(pb);
        delete pb;
    }
    set_gc_shape();
}

void
StructurePBGroup::delete_pseudobonds(const std::set<Pseudobond*>& pbs)
{
    delete_pbs_check(pbs);
    auto db = DestructionBatcher(this);
    for (auto pb: pbs) {
        _pbonds.erase(pb);
        delete pb;
    }
    set_gc_shape();
}

void
PBGroup::dtor_code()
{
    _destruction_relevant = false;
    auto du = DestructionUser(this);
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
    Pseudobond* pb = static_cast<Pseudobond*>(new CS_Pseudobond(a1, a2, this, cs));
    pb->set_color(color());
    pb->set_halfbond(halfbond());
    pb->set_radius(radius());
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
    pb->set_color(color());
    pb->set_halfbond(halfbond());
    pb->set_radius(radius());
    _pbonds.insert(pb); return pb;
}

const PBGroup::Pseudobonds&
CS_PBGroup::pseudobonds() const
{
    return pseudobonds(_structure->active_coord_set());
}

void
CS_PBGroup::remove_cs(const CoordSet* cs) {
    auto db = DestructionBatcher(this);
    for (auto pb: _pbonds[cs])
        delete pb;
    _pbonds.erase(cs);
}

std::pair<Atom*, Atom*>
StructurePBGroupBase::session_get_pb_ctor_info(int** ints) const
{
    auto& int_ptr = *ints;
    Atom* atoms[2];
    for (int i = 0; i < 2; ++i) {
        auto s = structure();
        if (s == nullptr) {
            // global pseudobond, need to determine structure for Atom
            auto ss_map = manager()->ses_id_to_struct_map();
            s = (*ss_map)[*int_ptr++];
        }
        atoms[i] = s->atoms()[*int_ptr++];
    }
    return std::pair<Atom*, Atom*>(atoms[0], atoms[1]);
}

void
StructurePBGroupBase::session_note_pb_ctor_info(Pseudobond* pb, int** ints) const
{
    auto& int_ptr = *ints;
    for (auto a: pb->atoms()) {
        auto s = structure();
        if (s == nullptr) {
            // note structure info
            s = a->structure();
            auto ss_map = manager()->ses_struct_to_id_map();
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
CS_PBGroup::session_num_floats(int version) const {
    int num_floats = SESSION_NUM_FLOATS(version) + StructurePBGroupBase::session_num_floats(version);
    for (auto crdset_pbs: _pbonds) {
        num_floats += crdset_pbs.second.size() * Pseudobond::session_num_floats(version);
    }
    return num_floats;
}

int
StructurePBGroup::session_num_floats(int version) const {
    return SESSION_NUM_FLOATS(version) + StructurePBGroupBase::session_num_floats(version)
        + pseudobonds().size() * Pseudobond::session_num_floats(version);
}

int
CS_PBGroup::session_num_ints(int version) const {
    int num_ints = SESSION_NUM_INTS(version) + StructurePBGroupBase::session_num_ints(version)
        + 2 * _pbonds.size(); // that last is for references to coord sets and # pbonds
    for (auto crdset_pbs: _pbonds) {
        // the +2 in the next line is for the atom IDs
        num_ints += crdset_pbs.second.size() * (Pseudobond::session_num_ints(version) + 2);
    }
    return num_ints;
}

int
StructurePBGroup::session_num_ints(int version) const {
    int num_ints = SESSION_NUM_INTS(version) + StructurePBGroupBase::session_num_ints(version)
        + pseudobonds().size() * (Pseudobond::session_num_ints(version) + 2); // +2 for the atom IDs
    if (structure() == nullptr) // will need to remember the structure IDs too
        num_ints += pseudobonds().size() * 2;
    return num_ints;
}

void
PBGroup::session_restore(int version, int** ints, float** floats)
{
    _color.session_restore(ints, floats);
    auto& int_ptr = *ints;
    _halfbond = int_ptr[0];
    int_ptr += SESSION_NUM_INTS(version);
    if (version > 6) {
        auto& float_ptr = *floats;
        _radius = float_ptr[0];
        float_ptr += SESSION_NUM_FLOATS(version);
    }
}

void
CS_PBGroup::session_restore(int version, int** ints, float** floats)
{
    StructurePBGroupBase::session_restore(version, ints, floats);
    auto& int_ptr = *ints;
    auto num_sets = int_ptr[0];
    int_ptr += SESSION_NUM_INTS(version);
    for (decltype(num_sets) i = 0; i < num_sets; ++i) {
        auto cs_index = *int_ptr++;
        auto cs = _structure->coord_sets()[cs_index];
        auto num_pbs = *int_ptr++;
        for (decltype(num_pbs) j = 0; j < num_pbs; ++j) {
            std::pair<Atom*, Atom*> atoms = session_get_pb_ctor_info(ints);
            auto pb = new_pseudobond(atoms.first, atoms.second, cs);
            pb->session_restore(version, ints, floats);
        }
    }
}

void
StructurePBGroup::session_restore(int version, int** ints, float** floats)
{
    StructurePBGroupBase::session_restore(version, ints, floats);
    auto& int_ptr = *ints;
    auto num_pbs = int_ptr[0];
    int_ptr += SESSION_NUM_INTS(version);
    for (decltype(num_pbs) i = 0; i < num_pbs; ++i) {
        std::pair<Atom*, Atom*> atoms = session_get_pb_ctor_info(ints);
        auto pb = new_pseudobond(atoms.first, atoms.second);
        pb->session_restore(version, ints, floats);
    }
}

void
PBGroup::session_save(int** ints, float** floats) const
{
    _color.session_save(ints, floats);
    auto& int_ptr = *ints;
    int_ptr[0] = _halfbond;
    int_ptr += SESSION_NUM_INTS();
    auto& float_ptr = *floats;
    float_ptr[0] = _radius;
    float_ptr += SESSION_NUM_FLOATS();
}

void
CS_PBGroup::session_save(int** ints, float** floats) const
{
    StructurePBGroupBase::session_save(ints, floats);
    auto& int_ptr = *ints;
    int_ptr[0] = _pbonds.size();
    int_ptr += SESSION_NUM_INTS();
    for (auto cs_pbs: _pbonds) {
        auto cs = cs_pbs.first;
        auto& pbs = cs_pbs.second;
        *int_ptr++ = (*structure()->session_save_crdsets)[cs];
        *int_ptr++ = pbs.size();
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
    auto& int_ptr = *ints;
    int_ptr[0] = _pbonds.size();
    int_ptr += SESSION_NUM_INTS();
    for (auto pb: _pbonds) {
        session_note_pb_ctor_info(pb, ints);
        pb->session_save(ints, floats);
    }
}

void
CS_PBGroup::session_save_setup() const
{
    auto ses_map = manager()->session_save_pbs;
    int id = ses_map->size();
    for (auto cs_pbs: _pbonds) {
        auto& pbs = cs_pbs.second;
        for (auto pb: pbs) {
            (*ses_map)[pb] = id++;
        }
    }
}

void
StructurePBGroup::session_save_setup() const
{
    auto ses_map = manager()->session_save_pbs;
    int id = ses_map->size();
    for (auto pb: _pbonds) {
        (*ses_map)[pb] = id++;
    }
}

}  // namespace atomstruct
