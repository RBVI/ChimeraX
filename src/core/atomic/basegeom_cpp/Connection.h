// vi: set expandtab ts=4 sw=4:
#ifndef basegeom_Connection
#define basegeom_Connection

#include <stdexcept>

#include "ChangeTracker.h"
#include "Graph.h"
#include "Real.h"
#include "Rgba.h"
#include "destruct.h"

// "forward declare" PyObject, which is a typedef of a struct,
// as per the python mailing list:
// http://mail.python.org/pipermail/python-dev/2003-August/037601.html
#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif
    
namespace basegeom {

using ::basegeom::ChangeTracker;

template <class End>
class Connection {
public:
    typedef End*  End_points[2];

    // Since this is a base class shared between Bond and Pseudobond,
    // the version number is specific to this class, rather than the
    // global number.  The conversion will be done by the derived class.
    // Therefore, update the derived class's session_base_version() when
    // this class's version changes
    static int  SESSION_NUM_INTS(int /*version*/=0) { return 3; }
    static int  SESSION_NUM_FLOATS(int /*version*/=0) { return 1; }
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
    void  session_info(bool intra_mol,
        PyObject* ints, PyObject* floats, PyObject* misc) const;
    Real  sqlength() const {
        return _end_points[0]->coord().sqdistance(_end_points[1]->coord());
    }

    // session related
    static int  session_num_floats(int version=0) {
        return SESSION_NUM_FLOATS(version) + Rgba::session_num_floats();
    }
    static int  session_num_ints(int version=0) {
        return SESSION_NUM_INTS(version) + Rgba::session_num_ints();
    }
    void  session_restore(int version, int** ints, float** floats) {
        _rgba.session_restore(ints, floats);
        auto& int_ptr = *ints;
        _display = int_ptr[0];
        _hide = int_ptr[1];
        _halfbond = int_ptr[2];
        int_ptr += SESSION_NUM_INTS(version);

        auto& float_ptr = *floats;
        _radius = float_ptr[0];
        float_ptr += SESSION_NUM_FLOATS(version);
    }
    void  session_save(int** ints, float** floats) const {
        _rgba.session_save(ints, floats);
        auto& int_ptr = *ints;
        int_ptr[0] = _display;
        int_ptr[1] = _hide;
        int_ptr[2] = _halfbond;
        int_ptr += SESSION_NUM_INTS();

        auto& float_ptr = *floats;
        float_ptr[0] = _radius;
        float_ptr += SESSION_NUM_FLOATS();
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
