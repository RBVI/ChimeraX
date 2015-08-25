// vi: set expandtab ts=4 sw=4:
#ifndef basegeom_Connection
#define basegeom_Connection

#include <stdexcept>

#include "Graph.h"
#include "Real.h"
#include "Rgba.h"
#include "destruct.h"

namespace basegeom {
    
template <class End>
class Connection {
public:
    enum class BondDisplay : unsigned char { Never, Smart, Always,
                                                MAX_VAL = Always };
    typedef End*  End_points[2];

protected:
    virtual const char*  err_msg_loop() const
        { return "Can't connect endpoint to itself"; }
    virtual const char*  err_msg_not_end() const
        { return "Endpoint arg of other_end() not in Connection"; }

    End_points  _end_points;

    BondDisplay  _display = BondDisplay::Smart;
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

    // graphics related
    const Rgba&  color() const { return _rgba; }
    BondDisplay  display() const { return _display; }
    bool  halfbond() const { return _halfbond; }
    virtual GraphicsContainer*  graphics_container() const = 0;
    void  set_color(Rgba::Channel r, Rgba::Channel g, Rgba::Channel b,
        Rgba::Channel a)
        { graphics_container()->set_gc_color(); _rgba = {r, g, b, a}; }
    void  set_color(const Rgba& rgba)
        { graphics_container()->set_gc_color(); _rgba = rgba; }
    void  set_display(BondDisplay d)
        { graphics_container()->set_gc_shape(); _display = d; }
    void  set_display(unsigned char d) { 
        if (d > static_cast<unsigned char>(BondDisplay::MAX_VAL))
            throw std::out_of_range("Invalid bond display value.");
        graphics_container()->set_gc_shape();
        _display = static_cast<BondDisplay>(d);
    }
    void  set_halfbond(bool hb)
        { graphics_container()->set_gc_color(); _halfbond = hb; }
    void  set_radius(float r)
        { graphics_container()->set_gc_shape(); _radius = r; }
    float  radius() const { return _radius; }
    int  hide() const { return _hide; }
    void  set_hide(int h)
        { graphics_container()->set_gc_shape(); _hide = h; }
    BondDisplay  visible() const
        { return _hide ? BondDisplay::Never : _display; }
};

template <class End, class FinalConnection>
class UniqueConnection: public Connection<End> {
protected:
    virtual const char*  err_msg_exists() const
        { return "Connection already exists between endpoints"; }
public:
    UniqueConnection(End *e1, End *e2);
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

template <class End, class FinalConnection>
UniqueConnection<End, FinalConnection>::UniqueConnection(End *e1, End *e2) :
    Connection<End>(e1, e2)
{
}

template <class End, class FinalConnection>
void
UniqueConnection<End, FinalConnection>::finish_construction()
{
    static_cast<Connection<End> *>(this)->finish_construction();
    End* e1 = this->_end_points[0]; // "this->" necessary because compiler
    End* e2 = this->_end_points[1]; // doesn't automatically look in parents
    if (e1->connects_to(e2))
        throw std::invalid_argument(err_msg_exists());
    e1->add_connection(static_cast<FinalConnection *>(this));
    e2->add_connection(static_cast<FinalConnection *>(this));
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
