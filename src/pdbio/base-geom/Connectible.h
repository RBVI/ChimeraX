#ifndef base_geom_Connectible
#define base_geom_Connectible

#include "Coord.h"

class Connectible {
public:
	virtual const Coord &  coord() const = 0;
	virtual void  set_coord(const Point & coord) = 0;
};
#endif  // base_geom_Connectible
