// vi: set expandtab ts=4 sw=4:
#include "destruct.h"

namespace basegeom {

void*  DestructionCoordinator::_destruction_parent = nullptr;
std::set<DestructionObserver*>  DestructionCoordinator::_observers;
std::set<void*>  DestructionCoordinator::_destroyed;

}  // namespace basegeom
