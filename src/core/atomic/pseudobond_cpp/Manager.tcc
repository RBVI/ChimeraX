// vi: set expandtab ts=4 sw=4:
#ifndef pseudobond_Manager_tcc
#define pseudobond_Manager_tcc

#include "Manager.h"

namespace pseudobond {

template <class Owner, class Grp_Class>
Owned_Manager<Owner, Grp_Class>::Owned_Manager(Owner* owner):
    Base_Manager<Grp_Class>(owner->change_tracker()), _owner(owner) {}

}

#endif  // pseudobond_Manager_tcc
