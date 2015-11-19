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

}  // namespace atomstruct
