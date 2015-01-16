// vi: set expandtab ts=4 sw=4:
#ifndef basegeom_Connectible
#define basegeom_Connectible

#include <algorithm>
#include <vector>

#include "Coord.h"
#include "Connection.h"
#include "Rgba.h"

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

    bool  _display = true;
    Rgba  _rgba;
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

    // graphics related
    const Rgba&  get_color() const { return _rgba; }
    bool  get_display() const { return _display; }
    void  set_color(Rgba::Channel r, Rgba::Channel g, Rgba::Channel b,
        Rgba::Channel a) { _rgba = {r, g, b, a}; }
    void  set_color(const Rgba& rgba) { _rgba = rgba; }
    void  set_display(bool d) { _display = d; }
};

} //  namespace basegeom

#endif  // basegeom_Connectible
