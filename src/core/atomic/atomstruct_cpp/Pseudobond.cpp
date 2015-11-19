// vi: set expandtab ts=4 sw=4:

#include "Atom.h"
#include "Pseudobond.h"

namespace atomstruct {

basegeom::ChangeTracker*
Pseudobond::change_tracker() const { return atoms()[0]->change_tracker(); }

}  // namespace atomstruct
