// vi: set expandtab ts=4 sw=4:
#ifndef basegeom_Connection
#define basegeom_Connection

#include <stdexcept>

#include "Real.h"
#include "Rgba.h"

namespace basegeom {
    
template <class End>
class Connection {
public:
    typedef End*  End_points[2];

protected:
    virtual const char*  err_msg_loop() const
        { return "Can't connect endpoint to itself"; }
    virtual const char*  err_msg_not_end() const
        { return "Endpoint arg of other_end() not in Connection"; }
private:
    End_points  _end_points;

    bool  _display = true;
    bool  _halfbond = true;
    float  _radius = 1.0;
    Rgba  _rgba;
public:
    Connection(End *e1, End *e2);
    virtual  ~Connection() {}
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
    bool  display() const { return _display; }
    bool  halfbond() const { return _halfbond; }
    void  set_color(Rgba::Channel r, Rgba::Channel g, Rgba::Channel b,
        Rgba::Channel a) { _rgba = {r, g, b, a}; }
    void  set_color(const Rgba& rgba) { _rgba = rgba; }
    void  set_display(bool d) { _display = d; }
    void  set_halfbond(bool hb) { _halfbond = hb; }
    void  set_radius(float r) { _radius = r; }
    float  radius() const { return _radius; }
};

template <class End, class FinalConnection>
class UniqueConnection: public Connection<End> {
protected:
    virtual const char*  err_msg_exists() const
        { return "Connection already exists between endpoints"; }
public:
    UniqueConnection(End *e1, End *e2);
    virtual  ~UniqueConnection() {}
};

template <class End>
Connection<End>::Connection(End *e1, End *e2)
{
    if (e1 == e2)
        throw std::invalid_argument(err_msg_loop());
    _end_points[0] = e1;
    _end_points[1] = e2;
}

template <class End, class FinalConnection>
UniqueConnection<End, FinalConnection>::UniqueConnection(End *e1, End *e2) :
    Connection<End>(e1, e2)
{
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
