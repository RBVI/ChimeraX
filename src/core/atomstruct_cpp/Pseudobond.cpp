// vim: set expandtab ts=4 sw=4:

#include "Atom.h"
#include "AtomicStructure.h"
#include "Pseudobond.h"

namespace pseudobond {
	
template <>
atomstruct::PBManager Global_Manager<atomstruct::PBGroup>::_manager = Global_Manager<atomstruct::PBGroup>();

}  // namespace pseudobond

namespace atomstruct {

void
Owned_PBGroup_Base::_check_ownership(Atom* a1, Atom* a2)
{
    if (a1->structure() != _owner || a2->structure() != _owner)
        throw std::invalid_argument("Pseudobond endpoints not in "
            " atomic structure associated with group");
}

Proxy_PBGroup*
AS_PBManager::get_group(const std::string& name, int create) const
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

    grp = new Proxy_PBGroup(name, _owner, create);
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
    PBond* pb = new PBond(a1, a2);
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
