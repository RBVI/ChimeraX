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


PBond*
CS_PBGroup::newPseudoBond(Atom* a1, Atom* a2)
{
    return newPseudoBond(a1, a2, a1->structure()->active_coord_set());
}

PBond*
CS_PBGroup::newPseudoBond(Atom* a1, Atom* a2, CoordSet* cs)
{
    _check_ownership(a1, a2);
    PBond* pb = makeLink(a1, a2);
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

}  // namespace atomstruct
