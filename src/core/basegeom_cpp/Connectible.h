// vi: set expandtab ts=4 sw=4:
#ifndef basegeom_Connectible
#define basegeom_Connectible

#include <algorithm>
#include <vector>

#include "Coord.h"
#include "Connection.h"

namespace basegeom {
    
template <class FinalConnection, class FinalConnectible>
class Connectible {
    friend class UniqueConnection<FinalConnectible, FinalConnection>;
protected:
    typedef std::vector<FinalConnection*> Connections;
    typedef std::vector<FinalConnectible*> Neighbors;

private:
    Connections  _connections; // _connections/_neighbors in same order
    Neighbors  _neighbors; // _connections/_neighbors in same order

protected:
    void  add_connection(FinalConnection *c) {
        _connections.push_back(c);
        _neighbors.push_back(
            c->other_end(static_cast<FinalConnectible *>(this)));
    }
    virtual  ~Connectible() {}
    const Connections&  connections() const { return _connections; }
    void  remove_connection(FinalConnection *c) {
        auto cnti = std::find(_connections.begin(), _connections.end(), c);
        _neighbors.erase(_neighbors.begin() + (cnti - _connections.begin()));
        _connections.erase(cnti);
    }
public:
    bool  connects_to(FinalConnectible *c) const {
        return std::find(_neighbors.begin(), _neighbors.end(), c)
            != _neighbors.end();
    }
    virtual const Coord &  coord() const = 0;
    const Neighbors&  neighbors() const { return _neighbors; }
    virtual void  set_coord(const Point & coord) = 0;
};

} //  namespace basegeom

#endif  // basegeom_Connectible
