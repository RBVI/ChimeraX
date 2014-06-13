// vim: set expandtab ts=4 sw=4:
#ifndef basegeom_Connection
#define basegeom_Connection

#include "Real.h"
#include <stdexcept>

namespace basegeom {
    
template <class End, class FinalConnection>
class Connection {
public:
    typedef End *  End_points[2];

private:
	End_points  _end_points;

public:
    Connection(End *e1, End *e2, const char *err1 = "Can't connect endpoint to itself",
        const char *err2 = "Connection already exists between endpoints");
    virtual  ~Connection() {}
    const End_points &  end_points() const { return _end_points; }
    End *  other_end(End *e,
        const char *err = "Endpoint arg of other_end() not in Connection") const;
    Real  sqlength() const {
        return _end_points[0]->coord().sqdistance(_end_points[1]->coord());
    }
};

template <class End, class FinalConnection>
Connection<End, FinalConnection>::Connection(End *e1, End *e2, const char *err1, const char *err2)
{
    if (e1 == e2)
        throw std::invalid_argument(err1);
    if (e1->connects_to(e2))
        throw std::invalid_argument(err2);
    _end_points[0] = e1;
    _end_points[1] = e2;
    e1->add_connection(static_cast<FinalConnection *>(this));
    e2->add_connection(static_cast<FinalConnection *>(this));
}

template <class End, class FinalConnection>
End *
Connection<End, FinalConnection>::other_end(End *e, const char *err) const
{
    if (e == _end_points[0])
        return _end_points[1];
    if (e == _end_points[1])
        return _end_points[0];
    throw std::invalid_argument(err);
}

} //  namespace basegeom

#endif  // basegeom_Connection
