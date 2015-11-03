// vi: set expandtab ts=4 sw=4:
#ifndef basegeom_Connectible
#define basegeom_Connectible

#include <algorithm>
#include <vector>

#include "ChangeTracker.h"
#include "Connection.h"
#include "Coord.h"
#include "Graph.h"
#include "Rgba.h"
#include "destruct.h"

namespace basegeom {

using ::basegeom::ChangeTracker;
using ::basegeom::Coord;
using ::basegeom::Point;
using ::basegeom::UniqueConnection;

template <class FinalGraph, class FinalConnectible, class FinalConnection>
class Connectible {
protected:
    friend class UniqueConnection<FinalConnectible, FinalConnection>;
    typedef std::vector<FinalConnection*> Connections;
    typedef std::vector<FinalConnectible*> Neighbors;

private:
    Connections  _connections; // _connections/_neighbors in same order
    FinalGraph*  _graph;
    Neighbors  _neighbors; // _connections/_neighbors in same order

    bool  _display = true;
    int  _hide = 0;
    bool  _selected = false;
    Rgba  _rgba;
public:
    Connectible(FinalGraph* graph) : _graph(graph) {}
    virtual  ~Connectible() { DestructionUser(this); }
    void  add_connection(FinalConnection *c);
    const Connections&  connections() const { return _connections; }
    bool  connects_to(FinalConnectible *c) const {
        return std::find(_neighbors.begin(), _neighbors.end(), c)
            != _neighbors.end();
    }
    virtual const Coord &  coord() const = 0;
    virtual FinalGraph*  graph() const { return _graph; }
    const Neighbors&  neighbors() const { return _neighbors; }
    void  remove_connection(FinalConnection *c);
    virtual void  set_coord(const Point & coord) = 0;

    // change tracking
    virtual ChangeTracker*  change_tracker() const = 0;

    // graphics related
    const Rgba&  color() const { return _rgba; }
    void  set_color(Rgba::Channel r, Rgba::Channel g, Rgba::Channel b, Rgba::Channel a) {
        set_color(Rgba({r, g, b, a}));
    }
    void  set_color(const Rgba& rgba) {
        if (rgba == _rgba)
            return;
        graphics_container()->set_gc_color();
        change_tracker()->add_modified(dynamic_cast<FinalConnectible*>(this),
            ChangeTracker::REASON_COLOR);
        _rgba = rgba;
    }
    bool  display() const { return _display; }
    virtual GraphicsContainer*  graphics_container() const = 0;
    void  set_display(bool d) {
        if (d == _display)
            return;
        graphics_container()->set_gc_shape();
        change_tracker()->add_modified(dynamic_cast<FinalConnectible*>(this),
            ChangeTracker::REASON_DISPLAY);
        _display = d;
    }
    int  hide() const { return _hide; }
    void  set_hide(int h) {
        if (h == _hide)
            return;
        if (h) {
            if (!_hide)
                graphics_container()->set_gc_shape();
        }
        else {
            if (_hide)
                graphics_container()->set_gc_shape();
        }
        change_tracker()->add_modified(dynamic_cast<FinalConnectible*>(this),
            ChangeTracker::REASON_HIDE);
        _hide = h;
    }
    bool  visible() const { return _display && !_hide; }
    bool  selected() const { return _selected; }
    void  set_selected(bool s) {
        if (s == _selected)
            return;
        graphics_container()->set_gc_select();
        change_tracker()->add_modified(dynamic_cast<FinalConnectible*>(this),
            ChangeTracker::REASON_SELECTED);
        _selected = s;
    }
};

} //  namespace basegeom

#endif  // basegeom_Connectible
