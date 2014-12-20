// vim: set expandtab ts=4 sw=4:
#ifndef templates_Bond
#define    templates_Bond

#include "../imex.h"

namespace tmpl {

class Atom;
class Molecule;

class ATOMSTRUCT_IMEX Bond {
    friend class Atom;
    friend class Molecule;
    void    operator=(const Bond &);    // disable
        Bond(const Bond &);    // disable
        ~Bond();
public:
    typedef Atom *    Atoms[2];
private:
    Atoms    _atoms;
public:
    const Atoms    &atoms() const { return _atoms; }
    Atom        *other_atom(const Atom *a) const;
private:
    Bond(Molecule *, Atom *a0, Atom *a1);
};

}  // namespace tmpl

#endif  // templates_Bond
