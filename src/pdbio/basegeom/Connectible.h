// vim: set expandtab ts=4 sw=4:
#ifndef basegeom_Connectible
#define basegeom_Connectible

#include "Coord.h"
#include <vector>
#include <map>
#include "Connection.h"

namespace basegeom {
    
template <class FinalConnection, class FinalConnectible>
class Connectible {
    friend class Connection<FinalConnectible, FinalConnection>;
protected:
    typedef std::map<FinalConnectible *, FinalConnection *> ConnectionsMap;
    typedef std::vector<FinalConnection *> Connections;

private:
    ConnectionsMap  _connections;

protected:
    void  add_connection(FinalConnection *c) {
        _connections[c->other_end(static_cast<FinalConnectible *>(this))] = c;
    }
    virtual  ~Connectible() {}
    Connections  connections() const;
    const ConnectionsMap &  connections_map() const { return _connections; }
    void  remove_connection(FinalConnection *c) {
        _connections.erase(c->other_end(static_cast<FinalConnectible *>(this)));
    }
public:
    bool  connects_to(FinalConnectible *c) const {
        return _connections.find(c) != _connections.end();
    }
    virtual const Coord &  coord() const = 0;
    virtual void  set_coord(const Point & coord) = 0;
};

template <class FinalConnection, class FinalConnectible>
typename Connectible<FinalConnection, FinalConnectible>::Connections
Connectible<FinalConnection, FinalConnectible>::connections() const
{
    std::vector<FinalConnection *> result;
    result.reserve(_connections.size());
    for (typename ConnectionsMap::const_iterator cmi = _connections.begin();
            cmi != _connections.end(); ++cmi) {
        result.push_back(cmi->second);
    }
    return result;
}

} //  namespace basegeom

#endif  // basegeom_Connectible
