// vim: set expandtab ts=4 sw=4:

#include "Pseudobond.h"
#include "Atom.h"

namespace pseudobond {
	
template <>
atomstruct::PBManager Global_Manager<atomstruct::PBGroup>::_manager = Global_Manager<atomstruct::PBGroup>();

}  // namespace pseudobond

namespace atomstruct {

PBond*
Owned_PBGroup_Base::newPseudoBond(Atom* a1, Atom* a2)
{
    if (a1->structure() != _owner || a2->structure() != _owner)
        throw std::invalid_argument("Pseudobond endpoints not in "
            " atomic structure associated with group");
    return addPseudoBond(makeLink(a1, a2));
}

}  // namespace atomstruct
