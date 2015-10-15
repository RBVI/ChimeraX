// vi: set expandtab ts=4 sw=4:
#ifndef basegeom_Sphere
#define basegeom_Sphere

#include "Connectible.h"
#include "Coord.h"

namespace basegeom {
    
template <class FinalConnectible, class FinalConnection>
class BaseSphere: public Connectible<FinalConnectible, FinalConnection> {
public:
    enum class DrawMode : unsigned char { Sphere, EndCap, Ball };
private:
    float  _radius;

    DrawMode  _draw_mode = DrawMode::Sphere;
public:
    BaseSphere(float radius): _radius(radius) {}
    virtual  ~BaseSphere() {}

    virtual const Coord&  coord() const = 0;
    virtual void  set_coord(const Point& coord) = 0;
    virtual void  set_radius(float r) {
        if (r == _radius)
            return;
        this->graphics_container()->set_gc_shape();
        this->change_tracker()->add_modified(dynamic_cast<FinalConnection*>(this),
            ChangeTracker::REASON_RADIUS);
        _radius = r;
    }
    virtual float  radius() const { return _radius; }

    // graphics related
    DrawMode  draw_mode() const { return _draw_mode; }
    void  set_draw_mode(DrawMode dm) {
        if (dm == _draw_mode)
            return;
        this->graphics_container()->set_gc_shape();
        this->change_tracker()->add_modified(dynamic_cast<FinalConnection*>(this),
            ChangeTracker::REASON_DRAW_MODE);
        _draw_mode = dm;
    }
};

template <class FinalConnectible, class FinalConnection>
class Sphere: public BaseSphere<FinalConnectible, FinalConnection> {
private:
    Coord  _coord;

public:
    virtual const Coord&  coord() const { return _coord; }
    virtual void  set_coord(const Point& coord) { _coord = coord; }
    virtual  ~Sphere() {}
};

} //  namespace basegeom

#endif  // basegeom_Sphere
