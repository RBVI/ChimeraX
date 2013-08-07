#ifndef molecule_Residue
#define molecule_Residue

#include <vector>
#include <map>
#include <string>

class Atom;
class Molecule;

class Residue {
	friend class Molecule;
public:
	typedef std::vector<Atom *>  Atoms;
	typedef std::multimap<std::string, Atom *>  AtomsMap;
private:
	Residue(Molecule *m, std::string &name, std::string &chain, int pos, char insert);
	Atoms  _atoms;
	std::string  _chain_id;
	char  _insertion_code;
	bool  _is_helix;
	bool  _is_het;
	bool  _is_sheet;
	Molecule *  _molecule;
	std::string  _name;
	int  _position;
	int  _ss_id;
public:
	void  add_atom(Atom *);
	const Atoms &  atoms() const { return _atoms; }
	AtomsMap  atoms_map() const;
	const std::string &  chain_id() const { return _chain_id; }
	int  count_atom(const std::string &) const;
	int  count_atom(const char *) const;
	Atom *  find_atom(const std::string &) const;
	Atom *  find_atom(const char *) const;
	char  insertion_code() const { return _insertion_code; }
	bool  is_het() const { return _is_het; }
	const std::string &  name() { return _name; }
	int  position() const { return _position; }
	void  set_is_helix(bool ih) { _is_helix = ih; }
	void  set_is_het(bool ih) { _is_het = ih; }
	void  set_is_sheet(bool is) { _is_sheet = is; }
	void  set_ss_id(int ssid) { _ss_id = ssid; }
	std::string  str() const;
};

#endif  // molecule_Residue
