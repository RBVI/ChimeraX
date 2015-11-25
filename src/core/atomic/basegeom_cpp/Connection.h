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

template <class End>
class Connection {
public:
    typedef End*  End_points[2];

    static const int  SESSION_NUM_INTS = 5;
    static const int  SESSION_NUM_FLOATS = 1;
protected:
    virtual const char*  err_msg_loop() const
        { return "Can't connect endpoint to itself"; }
    virtual const char*  err_msg_not_end() const
        { return "Endpoint arg of other_end() not in Connection"; }

    End_points  _end_points;

    bool  _display = true;
    int  _hide = 0;
    bool  _halfbond = true;
    float  _radius = 0.2;
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

    // session related
    virtual void  session_note_atoms(int** ints) const = 0;
    virtual void  session_note_structures(int** ) const {}
    static int  session_num_floats(bool /*global*/ = false) {
        return SESSION_NUM_FLOATS + Rgba::session_num_floats();
    }
    static int  session_num_ints(bool global = false) {
        return SESSION_NUM_INTS + Rgba::session_num_ints() + (global ? 2 : 0);
    }
    void  session_save(int** ints, float** floats, bool global = false) const {
        if (global) session_note_structures(ints);
        session_note_atoms(ints);
        _rgba.session_save(ints, floats);
        auto int_ptr = *ints;
        int_ptr[0] = _display;
        int_ptr[1] = _hide;
        int_ptr[2] = _halfbond;
        int_ptr += SESSION_NUM_INTS;

        auto float_ptr = *floats;
        float_ptr[0] = _radius;
        float_ptr += SESSION_NUM_FLOATS;
    }

    // change tracking
    virtual void  track_change(const std::string& reason) const = 0;

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
        track_change(ChangeTracker::REASON_COLOR);
        _rgba = rgba;
    }
    void  set_display(bool d) {
        if (d == _display)
            return;
        graphics_container()->set_gc_shape();
        track_change(ChangeTracker::REASON_DISPLAY);
        _display = d;
    }
    void  set_halfbond(bool hb) {
        if (hb == _halfbond)
            return;
        graphics_container()->set_gc_color();
        track_change(ChangeTracker::REASON_HALFBOND);
        _halfbond = hb;
    }
    void  set_radius(float r) {
        if (r == _radius)
            return;
        graphics_container()->set_gc_shape();
        track_change(ChangeTracker::REASON_RADIUS);
        _radius = r;
    }
    float  radius() const { return _radius; }
    int  hide() const { return _hide; }
    void  set_hide(int h) {
        if (h == _hide)
            return;
        graphics_container()->set_gc_shape();
        track_change(ChangeTracker::REASON_HIDE);
        _hide = h;
    }
    virtual bool shown() const
        { return visible() && _end_points[0]->visible() && _end_points[1]->visible(); }
    bool  visible() const
        { return _hide ? false : _display; }
};

template <class End>
class UniqueConnection: public Connection<End> {
protected:
    virtual const char*  err_msg_exists() const
        { return "Connection already exists between endpoints"; }
public:
    UniqueConnection(End *e1, End *e2);
    virtual void  add_to_endpoints() = 0;
    void  finish_construction(); // virtual calls now working...
    virtual  ~UniqueConnection() {}
};

template <class End>
Connection<End>::Connection(End *e1, End *e2)
{
    _end_points[0] = e1;
    _end_points[1] = e2;
}

template <class End>
void
Connection<End>::finish_construction()
{
    if (_end_points[0] == _end_points[1])
        throw std::invalid_argument(err_msg_loop());
    graphics_container()->set_gc_shape();
}

template <class End>
UniqueConnection<End>::UniqueConnection(End *e1, End *e2) :
    Connection<End>(e1, e2)
{
}

template <class End>
void
UniqueConnection<End>::finish_construction()
{
    static_cast<Connection<End> *>(this)->finish_construction();
    End* e1 = this->_end_points[0]; // "this->" necessary because compiler
    End* e2 = this->_end_points[1]; // doesn't automatically look in parents
    if (e1->connects_to(e2))
        throw std::invalid_argument(err_msg_exists());
    add_to_endpoints();
}

template <class End>
End *
Connection<End>::other_end(End *e) const
{
    if (e == _end_points[0])
        return _end_points[1];
    if (e == _end_points[1])
        return _end_points[0];
    throw std::invalid_argument(err_msg_not_end());
}

} //  namespace basegeom

#endif  // basegeom_Connection
