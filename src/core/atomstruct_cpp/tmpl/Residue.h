// vi: set expandtab ts=4 sw=4:
#ifndef templates_Residue
#define    templates_Residue

#include <map>
#include <vector>
#include "../imex.h"
#include "../string_types.h"

namespace tmpl {

using atomstruct::AtomName;
using atomstruct::AtomType;
using atomstruct::ResName;

class Atom;
class Molecule;

class ATOMSTRUCT_IMEX Residue {
    friend class Molecule;
    void    operator=(const Residue &);    // disable
    Residue(const Residue &);    // disable
    ~Residue();
public:
    void    add_atom(Atom *element);
    Atom    *find_atom(const AtomName&) const;

    // return atoms that received assignments from the template
    std::vector<Atom *>    template_assign(
                  void (Atom::*assign_func)(const AtomType&),
                  const char *app,
                  const char *template_dir,
                  const char *extension
                ) const;

    const ResName&    name() const { return _name; }
    Atom    *chief() const { return _chief; }
    void    chief(Atom *a) { _chief = a; }
    Atom    *link() const { return _link; }
    void    link(Atom *a) { _link = a; }
    std::string    description() const { return _description; }
    void    description(const std::string &d) { _description = d; }

    // alternative to chief/link for mmCIF templates
    typedef std::vector<Atom *> AtomList;
    void    add_link_atom(Atom *);
    const AtomList link_atoms() const { return _link_atoms; }

    typedef std::map<AtomName, Atom *> AtomsMap;
    const AtomsMap &atoms_map() { return _atoms; }
private:
    Residue(Molecule *, const char *t);
    ResName     _name;
    AtomsMap    _atoms;
    Atom        *_chief, *_link;
    std::string _description;
    AtomList    _link_atoms;
};

}  // namespace tmpl

#endif  // templates_Residue
