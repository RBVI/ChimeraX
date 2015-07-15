// vi: set expandtab ts=4 sw=4:
#ifndef basegeom_Connectible
#define basegeom_Connectible

#include <algorithm>
#include <vector>

#include "Connection.h"
#include "Coord.h"
#include "Graph.h"
#include "Rgba.h"
#include "destruct.h"

namespace basegeom {

using ::basegeom::Coord;
using ::basegeom::Point;
using ::basegeom::UniqueConnection;
    
template <class FinalConnection, class FinalConnectible>
class Connectible {
protected:
    friend class UniqueConnection<FinalConnectible, FinalConnection>;
    typedef std::vector<FinalConnection*> Connections;
    typedef std::vector<FinalConnectible*> Neighbors;

private:
    Connections  _connections; // _connections/_neighbors in same order
    Neighbors  _neighbors; // _connections/_neighbors in same order

    bool  _display = true;
    bool  _selected = false;
    Rgba  _rgba;
public:
    virtual  ~Connectible() { DestructionUser(this); }
    void  add_connection(FinalConnection *c);
    const Connections&  connections() const { return _connections; }
    bool  connects_to(FinalConnectible *c) const {
        return std::find(_neighbors.begin(), _neighbors.end(), c)
            != _neighbors.end();
    }
    virtual const Coord &  coord() const = 0;
    const Neighbors&  neighbors() const { return _neighbors; }
    void  remove_connection(FinalConnection *c);
    virtual void  set_coord(const Point & coord) = 0;

    // graphics related
    const Rgba&  color() const { return _rgba; }
    void  set_color(Rgba::Channel r, Rgba::Channel g, Rgba::Channel b,
        Rgba::Channel a)
        { graphics_container()->set_gc_redraw(); _rgba = {r, g, b, a}; }
    void  set_color(const Rgba& rgba)
        { graphics_container()->set_gc_redraw(); _rgba = rgba; }
    bool  display() const { return _display; }
    virtual GraphicsContainer*  graphics_container() const = 0;
    void  set_display(bool d)
        { graphics_container()->set_gc_shape(); _display = d; }
    bool  selected() const { return _selected; }
    void  set_selected(bool s)
        { graphics_container()->set_gc_select(); _selected = s; }
};

} //  namespace basegeom

#endif  // basegeom_Connectible
