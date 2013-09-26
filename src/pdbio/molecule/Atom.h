#ifndef molecule_Atom
#define molecule_Atom

#include <vector>
#include <string>
#include <map>

#include "Element.h"
#include "Point.h"
#include "Coord.h"
#include "Bond.h"

class CoordSet;
class Molecule;
class Residue;

class Atom {
	friend class Molecule;
	friend class Residue;
public:
	typedef std::map<Atom *, Bond *> BondsMap;
	typedef std::vector<Bond *> Bonds;

private:
	static const unsigned int  COORD_UNASSIGNED = ~0u;
	Atom(Molecule *m, std::string &name, Element e);
	BondsMap  _bonds;
	unsigned int  _coord_index;
	Element  _element;
	Molecule *  _molecule;
	std::string  _name;
	Residue *  _residue;
	typedef struct {
		std::vector<float> *  aniso_u;
		float  bfactor;
		Point  coord;
		float  occupancy;
		int  serial_number;
	} _Alt_loc_info;
	typedef std::map<char, _Alt_loc_info>  _Alt_loc_map;
	_Alt_loc_map  _alt_loc_map;
	char  _alt_loc;
	std::vector<float> *  _aniso_u;
	int  _serial_number;
	void  _coordset_set_coord(const Point &);
	void  _coordset_set_coord(const Point &, CoordSet *cs);
	unsigned int  _new_coord(const Point &);

public:
	void  add_bond(Bond *b) { _bonds[b->other_atom(this)] = b; }
	float  bfactor() const;
	Bonds  bonds() const;
	const BondsMap &	bonds_map() const { return _bonds; }
	bool  connects_to(Atom *a) const {
		return _bonds.find(a) != _bonds.end();
	}
	unsigned int  coord_index() const { return _coord_index; }
	const Coord &coord() const;
	Element  element() const { return _element; }
	Molecule *  molecule() const { return _molecule; }
	const std::string  name() const { return _name; }
	float  occupancy() const;
	void  register_field(std::string name, int value) {}
	void  register_field(std::string name, double value) {}
	void  register_field(std::string name, const std::string value) {}
	void  remove_bond(Bond *);
	Residue *  residue() const { return _residue; }
	void  set_alt_loc(char alt_loc, bool create=false);
	void  set_aniso_u(float u11, float u12, float u13, float u22, float u23, float u33);
	void  set_bfactor(float);
	void  set_coord(const Point & coord, CoordSet * cs=NULL);
	void  set_occupancy(float);
	void  set_serial_number(int);
};

#endif  // molecule_Atom
