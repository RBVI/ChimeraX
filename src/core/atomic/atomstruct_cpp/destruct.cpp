// vi: set expandtab ts=4 sw=4:
#define ATOMSTRUCT_EXPORT
#include "destruct.h"

namespace atomstruct {

ATOMSTRUCT_IMEX void*  DestructionCoordinator::_destruction_batcher = nullptr;
ATOMSTRUCT_IMEX void*  DestructionCoordinator::_destruction_parent = nullptr;
ATOMSTRUCT_IMEX std::set<DestructionObserver*>  DestructionCoordinator::_observers;
ATOMSTRUCT_IMEX std::set<void*>  DestructionCoordinator::_destroyed;
ATOMSTRUCT_IMEX int DestructionCoordinator::_num_notifications_off = 0;

}  // namespace atomstruct
