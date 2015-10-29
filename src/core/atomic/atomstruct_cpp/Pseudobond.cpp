// vi: set expandtab ts=4 sw=4:

#include "Atom.h"
#include "AtomicStructure.h"
#include "Pseudobond.h"
#include <basegeom/destruct.h>
#include <pseudobond/Manager.tcc>

namespace pseudobond {

// force instantiation
template class Owned_Manager<atomstruct::AtomicStructure, atomstruct::Proxy_PBGroup>;
//template Owned_Manager<atomstruct::AtomicStructure, atomstruct::Proxy_PBGroup>::Owned_Manager(atomstruct::Proxy_PBGroup*);

}

namespace atomstruct {

basegeom::ChangeTracker*
PBond::change_tracker() const { return atoms()[0]->change_tracker(); }

void
Owned_PBGroup_Base::_check_ownership(Atom* a1, Atom* a2)
{
    if (_owner != nullptr && (a1->structure() != _owner || a2->structure() != _owner))
        throw std::invalid_argument("Pseudobond endpoints not in "
            " atomic structure associated with group");
}

static void
_check_destroyed_atoms(PBonds& pbonds, const std::set<void*>& destroyed,
    GraphicsContainer* gc)
{
    PBonds remaining;
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
Owned_PBGroup::check_destroyed_atoms(const std::set<void*>& destroyed)
{
    auto db = basegeom::DestructionBatcher(this);
    _check_destroyed_atoms(_pbonds, destroyed,
        static_cast<GraphicsContainer*>(this));
}

void
AS_PBManager::delete_group(Proxy_PBGroup* group)
{
    auto gmi = this->_groups.find(group->category());
    if (gmi == this->_groups.end())
        throw std::invalid_argument("Asking for deletion of group not in manager!");
    delete group;
    this->_groups.erase(gmi);
}

void
PBManager::delete_group(Proxy_PBGroup* group)
{
    auto gmi = this->_groups.find(group->category());
    if (gmi == this->_groups.end())
        throw std::invalid_argument("Asking for deletion of group not in manager!");
    delete group;
    this->_groups.erase(gmi);
}

Proxy_PBGroup*
AS_PBManager::get_group(const std::string& name, int create)
{
    Proxy_PBGroup* grp;
    auto gmi = this->_groups.find(name);
    if (gmi != this->_groups.end()) {
        grp = (*gmi).second;
        if (create != GRP_NONE && grp->group_type() != create) {
            throw std::invalid_argument("Group type mismatch");
        }
        return grp;
    }

    if (create == GRP_NONE)
        return nullptr;

    grp = new Proxy_PBGroup(static_cast<Proxy_PBGroup::BaseManager*>(this),
        name, owner(), create);
    if (name == owner()->PBG_METAL_COORDINATION)
        grp->set_default_color(147, 112, 219);
    else if (name == owner()->PBG_MISSING_STRUCTURE)
        grp->set_default_halfbond(true);
    else if (name == owner()->PBG_HYDROGEN_BONDS)
        grp->set_default_color(0, 204, 230);
    _groups[name] = grp;
    return grp;
}

Proxy_PBGroup*
PBManager::get_group(const std::string& name, int create)
{
    Proxy_PBGroup* grp;
    auto gmi = this->_groups.find(name);
    if (gmi != this->_groups.end()) {
        grp = (*gmi).second;
        if (create != GRP_NONE && grp->group_type() != create) {
            throw std::invalid_argument("Group type mismatch");
        }
        return grp;
    }

    if (create == GRP_NONE)
        return nullptr;

    if (create != GRP_NORMAL)
        throw std::invalid_argument("Can only create normal pseudobond groups"
            " in global non-structure-associated pseudobond manager");

    grp = new Proxy_PBGroup(static_cast<Proxy_PBGroup::BaseManager*>(this),
        name, nullptr, create);
    _groups[name] = grp;
    return grp;
}

PBond*
CS_PBGroup::new_pseudobond(Atom* a1, Atom* a2)
{
    return new_pseudobond(a1, a2, a1->structure()->active_coord_set());
}

PBond*
CS_PBGroup::new_pseudobond(Atom* a1, Atom* a2, CoordSet* cs)
{
    _check_ownership(a1, a2);
    PBond* pb = new PBond(a1, a2, this);
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

const std::set<PBond*>&
CS_PBGroup::pseudobonds() const
{
    return pseudobonds(_owner->active_coord_set());
}

void
AS_PBManager::remove_cs(const CoordSet* cs) {
    for (auto pbg_info: _groups) pbg_info.second->remove_cs(cs);
}

}  // namespace atomstruct
