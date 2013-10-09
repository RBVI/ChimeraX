#ifndef base_geom_Sphere
#define base_geom_Sphere

#include "Connectible.h"
#include "Coord.h"

class BaseSphere: public Connectible {
private:
	float  _radius;
public:
	BaseSphere(): _radius(0.0) {}
	void  set_radius(float r) { _radius = r; }
	float  radius() const { return _radius; }
};

class Sphere: public BaseSphere {
private:
	Coord  _coord;

public:
	virtual const Coord &  coord() const { return _coord; }
	virtual void  set_coord(const Point & coord) { _coord = coord; }
};

#endif  // base_geom_Sphere
