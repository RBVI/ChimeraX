// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2016 Regents of the University of California.
 * All rights reserved.  This software provided pursuant to a
 * license agreement containing restrictions on its disclosure,
 * duplication and use.  For details see:
 * http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
 * This notice must be embedded in or attached to all copies,
 * including partial copies, of the software or any revisions
 * or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#ifndef element_Element
#define element_Element

// $Id: Element.h 36237 2012-04-26 00:02:50Z goddard $

#include <iostream>
#include <map>
#include <set>
#include <string>

#include "imex.h"

namespace element {

#if 0
class Element;
extern template class ELEMENT_IMEX std::map<int, const Element*>;
extern template class ELEMENT_IMEX std::set<std::string>;
extern template class ELEMENT_IMEX std::set<int>;
#endif

class ELEMENT_IMEX Element {
public:
    // Atomic Symbols:
    enum AS {
        LONE_PAIR, H, D = 1, T = 1, He,
        Li, Be, B, C, N, O, F, Ne,
        Na, Mg, Al, Si, P, S, Cl, Ar,
        K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn,
        Ga, Ge, As, Se, Br, Kr,
        Rb, Sr, Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd,
        In, Sn, Sb, Te, I, Xe,
        Cs, Ba, La,
        Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu,
        Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg, Tl, Pb, Bi, Po, At, Rn,
        Fr, Ra, Ac,
        Th, Pa, U, Np, Pu, Am, Cm, Bk, Cf, Es, Fm, Md, No, Lr,
        Rf, Db, Sg, Bh, Hs, Mt, Ds, Rg,
        Cn, Uut, Fl, Uup, Uuh, Uus, Uuo
    };
    static float    bond_radius(const Element&);
    static float    bond_length(const Element&, const Element&);
private:
    Element(AS a): as(a) {}
    // prevent copy/assignment
    Element(const Element&);
    Element& operator=(const Element&);

    static std::set<int>  _alkali_metals;
    static std::map<int, const Element*>  _elements;
    static std::set<int>  _halogens;
    static std::set<std::string>  _names;
    static std::set<int>  _noble_gases;

    static AS    atomic_number(const char *name);
    AS        as;        // atomic number
public:
    static const Element&  get_element(const char* name) {
        return get_element(atomic_number(name));
    }
    static const Element&  get_element(int i);

    long  hash() const { return number(); }
    bool  is_alkali_metal() const
        { return _alkali_metals.find(number()) != _alkali_metals.end(); }
    bool  is_halogen() const
        { return _halogens.find(number()) != _halogens.end(); }
    bool  is_metal() const;
    bool  is_noble_gas() const
        { return _noble_gases.find(number()) != _noble_gases.end(); }
    float  mass() const;        // standard atomic weight
    const char*  name() const;
    static const std::set<std::string>&  names();
    int  number() const { return int(as); }
    int  valence() const;

    bool  operator==(const Element &a) const { return as == a.as; }
    bool  operator!=(const Element &a) const { return as != a.as; }
    bool  operator<(const Element &a) const { return as < a.as; }
    bool  operator<=(const Element &a) const { return as <= a.as; }
    bool  operator>(const Element &a) const { return as > a.as; }
    bool  operator>=(const Element &a) const { return as >= a.as; }
    bool  operator==(AS sym) const { return as == sym; }
    bool  operator!=(AS sym) const { return as != sym; }
    bool  operator<(AS sym) const { return as < sym; }
    bool  operator<=(AS sym) const { return as <= sym; }
    bool  operator>(AS sym) const { return as > sym; }
    bool  operator>=(AS sym) const { return as >= sym; }
};

inline std::ostream &
operator<<(std::ostream &os, const Element &a)
{
    os << a.name();
    return os;
}

}  // namespace element

#endif  // element_Element
