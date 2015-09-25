// vi: set expandtab ts=4 sw=4:
#ifndef element_Element
#define element_Element

// Copyright (c) 1996-2009 The Regents of the University of California.
// All rights reserved.
// 
// Redistribution and use in source and binary forms are permitted
// provided that the above copyright notice and this paragraph are
// duplicated in all such forms and that any documentation,
// distribution and/or use acknowledge that the software was developed
// by the Computer Graphics Laboratory, University of California,
// San Francisco.  The name of the University may not be used to
// endorse or promote products derived from this software without
// specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
// WARRANTIES OF MERCHANTIBILITY AND FITNESS FOR A PARTICULAR PURPOSE.
// IN NO EVENT SHALL THE REGENTS OF THE UNIVERSITY OF CALIFORNIA BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
// OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF THIS SOFTWARE.

// $Id: Element.h 36237 2012-04-26 00:02:50Z goddard $

#include <iostream>
#include <map>
#include <set>

#include "imex.h"

namespace element {

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
    static std::set<int>  _halides;
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
    bool  is_halide() const
        { return _halides.find(number()) != _halides.end(); }
    bool  is_metal() const;
    bool  is_noble_gas() const
        { return _noble_gases.find(number()) != _noble_gases.end(); }
    float  mass() const;        // standard atomic weight
    const char*  name() const;
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
