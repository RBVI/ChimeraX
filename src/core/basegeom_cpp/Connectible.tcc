// vi: set expandtab ts=4 sw=4:
#ifndef basegeom_Connectible_tcc
#define basegeom_Connectible_tcc

#include <algorithm>

#include "Connectible.h"

namespace basegeom {

template <class FinalConnection, class FinalConnectible>
inline void
Connectible<FinalConnection, FinalConnectible>::add_connection(
    FinalConnection *c)
{
    _connections.push_back(c);
    _neighbors.push_back(
        c->other_end(static_cast<FinalConnectible *>(this)));
    graphics_container()->set_gc_shape();
}

template <class FinalConnection, class FinalConnectible>
inline void
Connectible<FinalConnection, FinalConnectible>::remove_connection(
    FinalConnection *c)
{
    auto cnti = std::find(_connections.begin(), _connections.end(), c);
    _neighbors.erase(_neighbors.begin() + (cnti - _connections.begin()));
    _connections.erase(cnti);
}

} //  namespace basegeom

#endif  // basegeom_Connectible_tcc
