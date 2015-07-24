// vi: set expandtab ts=4 sw=4:
#ifndef basegeom_Sphere
#define basegeom_Sphere

#include "Connectible.h"
#include "Coord.h"

namespace basegeom {
    
template <class FinalConnection, class FinalConnectible>
class BaseSphere: public Connectible<FinalConnection, FinalConnectible> {
private:
    float  _radius;

    int  _draw_mode = 0;
public:
    BaseSphere(float radius): _radius(radius) {}
    virtual  ~BaseSphere() {}
    virtual void  set_radius(float r)
        {   if (_radius != r) {
                this->graphics_container()->set_gc_shape();
                _radius = r;
            }
        }
    virtual float  radius() const { return _radius; }

    // graphics related
    int  draw_mode() const { return _draw_mode; }
    void  set_draw_mode(int dm)
        { this->graphics_container()->set_gc_shape(); _draw_mode = dm; }
};

template <class FinalConnection, class FinalConnectible>
class Sphere: public BaseSphere<FinalConnection, FinalConnectible> {
private:
    Coord  _coord;

public:
    virtual const Coord &  coord() const { return _coord; }
    virtual void  set_coord(const Point & coord) { _coord = coord; }
    virtual  ~Sphere() {}
};

} //  namespace basegeom

#endif  // basegeom_Sphere
