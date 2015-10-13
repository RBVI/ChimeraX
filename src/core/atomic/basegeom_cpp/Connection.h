// vi: set expandtab ts=4 sw=4:
#ifndef basegeom_Connection
#define basegeom_Connection

#include <stdexcept>

#include "ChangeTracker.h"
#include "Graph.h"
#include "Real.h"
#include "Rgba.h"
#include "destruct.h"

namespace basegeom {
    
using ::basegeom::ChangeTracker;

template <class End, class FinalConnection>
class Connection {
public:
    typedef End*  End_points[2];

protected:
    virtual const char*  err_msg_loop() const
        { return "Can't connect endpoint to itself"; }
    virtual const char*  err_msg_not_end() const
        { return "Endpoint arg of other_end() not in Connection"; }

    End_points  _end_points;

    bool  _display = true;
    int  _hide = 0;
    bool  _halfbond = true;
    float  _radius = 1.0;
    Rgba  _rgba;
public:
    Connection(End *e1, End *e2);
    void  finish_construction(); // virtual calls now working...
    virtual  ~Connection() { auto du = DestructionUser(this); }
    bool  contains(End* e) const {
        return e == _end_points[0] || e == _end_points[1];
    }
    const End_points &  end_points() const { return _end_points; }
    Real  length() const {
        return _end_points[0]->coord().distance(_end_points[1]->coord());
    }
    End *  other_end(End* e) const;
    Real  sqlength() const {
        return _end_points[0]->coord().sqdistance(_end_points[1]->coord());
    }

    // change tracking
    virtual ChangeTracker*  change_tracker() const = 0;

    // graphics related
    const Rgba&  color() const { return _rgba; }
    bool  display() const { return _display; }
    bool  halfbond() const { return _halfbond; }
    virtual GraphicsContainer*  graphics_container() const = 0;
    void  set_color(Rgba::Channel r, Rgba::Channel g, Rgba::Channel b, Rgba::Channel a)
        { set_color(Rgba({r, g, b, a})); }
    void  set_color(const Rgba& rgba) {
        if (rgba == _rgba)
            return;
        graphics_container()->set_gc_color();
        change_tracker()->add_modified(dynamic_cast<FinalConnection*>(this),
            ChangeTracker::REASON_COLOR);
        _rgba = rgba;
    }
    void  set_display(bool d) {
        if (d == _display)
            return;
        graphics_container()->set_gc_shape();
        change_tracker()->add_modified(dynamic_cast<FinalConnection*>(this),
            ChangeTracker::REASON_DISPLAY);
        _display = d;
    }
    void  set_halfbond(bool hb) {
        if (hb == _halfbond)
            return;
        graphics_container()->set_gc_color();
        change_tracker()->add_modified(dynamic_cast<FinalConnection*>(this),
            ChangeTracker::REASON_HALFBOND);
        _halfbond = hb;
    }
    void  set_radius(float r) {
        if (r == _radius)
            return;
        graphics_container()->set_gc_shape();
        change_tracker()->add_modified(dynamic_cast<FinalConnection*>(this),
            ChangeTracker::REASON_RADIUS);
        _radius = r;
    }
    float  radius() const { return _radius; }
    int  hide() const { return _hide; }
    void  set_hide(int h) {
        if (h == _hide)
            return;
        graphics_container()->set_gc_shape();
        change_tracker()->add_modified(dynamic_cast<FinalConnection*>(this),
            ChangeTracker::REASON_HIDE);
        _hide = h;
    }
    virtual bool shown() const
        { return visible() && _end_points[0]->visible() && _end_points[1]->visible(); }
    bool  visible() const
        { return _hide ? false : _display; }
};

template <class End, class FinalConnection>
class UniqueConnection: public Connection<End, FinalConnection> {
protected:
    virtual const char*  err_msg_exists() const
        { return "Connection already exists between endpoints"; }
public:
    UniqueConnection(End *e1, End *e2);
    void  finish_construction(); // virtual calls now working...
    virtual  ~UniqueConnection() {}
};

template <class End, class FinalConnection>
Connection<End, FinalConnection>::Connection(End *e1, End *e2)
{
    _end_points[0] = e1;
    _end_points[1] = e2;
}

template <class End, class FinalConnection>
void
Connection<End, FinalConnection>::finish_construction()
{
    if (_end_points[0] == _end_points[1])
        throw std::invalid_argument(err_msg_loop());
    graphics_container()->set_gc_shape();
}

template <class End, class FinalConnection>
UniqueConnection<End, FinalConnection>::UniqueConnection(End *e1, End *e2) :
    Connection<End, FinalConnection>(e1, e2)
{
}

template <class End, class FinalConnection>
void
UniqueConnection<End, FinalConnection>::finish_construction()
{
    static_cast<Connection<End, FinalConnection> *>(this)->finish_construction();
    End* e1 = this->_end_points[0]; // "this->" necessary because compiler
    End* e2 = this->_end_points[1]; // doesn't automatically look in parents
    if (e1->connects_to(e2))
        throw std::invalid_argument(err_msg_exists());
    e1->add_connection(dynamic_cast<FinalConnection *>(this));
    e2->add_connection(static_cast<FinalConnection *>(this));
}

template <class End, class FinalConnection>
End *
Connection<End, FinalConnection>::other_end(End *e) const
{
    if (e == _end_points[0])
        return _end_points[1];
    if (e == _end_points[1])
        return _end_points[0];
    throw std::invalid_argument(err_msg_not_end());
}

} //  namespace basegeom

#endif  // basegeom_Connection
