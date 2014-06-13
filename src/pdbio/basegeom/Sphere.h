// vim: set expandtab ts=4 sw=4:
#ifndef base_geom_Sphere
#define base_geom_Sphere

#include "Connectible.h"
#include "Coord.h"

template <class FinalConnection, class FinalConnectible>
class BaseSphere: public Connectible<FinalConnection, FinalConnectible> {
private:
    float  _radius;
public:
    BaseSphere(): _radius(0.0) {}
    virtual  ~BaseSphere() {}
    void  set_radius(float r) { _radius = r; }
    float  radius() const { return _radius; }
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

#endif  // base_geom_Sphere
