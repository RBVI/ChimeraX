// vi: set expandtab ts=4 sw=4:

#include "Atom.h"
#include "ChangeTracker.h"
#include "PBGroup.h"
#include "PBManager.h"
#include "Pseudobond.h"

namespace atomstruct {

ChangeTracker*
Pseudobond::change_tracker() const { return atoms()[0]->change_tracker(); }

GraphicsContainer*
Pseudobond::graphics_container() const { return static_cast<GraphicsContainer*>(group()); }

}  // namespace atomstruct
