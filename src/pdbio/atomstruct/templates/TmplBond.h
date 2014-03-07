// vim: set expandtab ts=4 sw=4:
#ifndef templates_TmplBond
#define    templates_TmplBond

#include "../imex.h"

class TmplAtom;
class TmplMolecule;

class ATOMSTRUCT_IMEX TmplBond {
    friend class TmplAtom;
    friend class TmplMolecule;
    void    operator=(const TmplBond &);    // disable
        TmplBond(const TmplBond &);    // disable
        ~TmplBond();
public:
    typedef TmplAtom *    Atoms[2];
private:
    Atoms    _atoms;
public:
    const Atoms    &atoms() const { return _atoms; }
    TmplAtom        *other_atom(const TmplAtom *a) const;
private:
    TmplBond(TmplMolecule *, TmplAtom *a0, TmplAtom *a1);
};

#endif  // templates_TmplBond
