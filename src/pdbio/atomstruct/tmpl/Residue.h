// vim: set expandtab ts=4 sw=4:
#ifndef templates_Residue
#define    templates_Residue

#include <map>
#include <vector>
#include "Residue_template.h"
#include "../imex.h"

namespace tmpl {

class Atom;
class Molecule;

class ATOMSTRUCT_IMEX Residue {
    friend class Molecule;
    void    operator=(const Residue &);    // disable
        Residue(const Residue &);    // disable
        ~Residue();
public:
    void    add_atom(Atom *element);
    typedef std::map<std::string, Atom *> AtomsMap;
    Atom    *find_atom(const std::string &) const;
public:
    inline const std::string    name() const { return _name; }
private:
    std::string    _name;
    AtomsMap    _atoms;
public:
    // return atoms that received assignments from the template
    std::vector<Atom *>    template_assign(
                  void (*assign_func)(Atom *, const char *),
                  const char *app,
                  const char *template_dir,
                  const char *extension
                ) const;
    std::vector<Atom *>    template_assign(
                  void (Atom::*assign_func)(const char *),
                  const char *app,
                  const char *template_dir,
                  const char *extension
                ) const;
    std::vector<Atom *>    template_assign(tmpl_assigner, 
                  const char *app,
                  const char *template_dir,
                  const char *extension
                ) const;
private:
    Atom    *_chief, *_link;
    std::string    _description;
public:
    Atom    *chief() const { return _chief; }
    void    chief(Atom *a) { _chief = a; }
    Atom    *link() const { return _link; }
    void    link(Atom *a) { _link = a; }
    std::string    description() const { return _description; }
    void        description(const std::string &d) { _description = d; }
private:
    Residue(Molecule *, const char *t);
};

}  // namespace tmpl

#endif  // templates_Residue
