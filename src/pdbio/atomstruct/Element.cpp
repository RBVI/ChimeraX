// vim: set expandtab ts=4 sw=4:
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

// $Id: Element.cpp 36237 2012-04-26 00:02:50Z goddard $

#include "Element.h"
#include <ctype.h>

namespace atomstruct {

char const * const symbols[] = {
    "LP",  "H", "He", "Li", "Be",  "B",  "C",  "N",  "O",
     "F", "Ne", "Na", "Mg", "Al", "Si",  "P",  "S", "Cl",
    "Ar",  "K", "Ca", "Sc", "Ti",  "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
    "Kr", "Rb", "Sr",  "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",  "I",
    "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm",
    "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta",  "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac",
    "Th", "Pa",  "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh",
    "Hs", "Mt", "Ds", "Rg", "Uub", "Uut", "Uuq", "Uup", "Uuh",
    "Uus", "Uuo"
};

const int NUM_SYMBOLS = sizeof symbols / sizeof symbols[0];

//    The following covalent bond radii are taken from: Elaine C. Meng
//    and Richard A. Lewis, "Determination of Molecular Topology and
//    Atomic Hybridization States from Heavy Atom Coordinates," Journal
//    of Computational Chemistry, Vol. 12, No. 7, pp. 891-898 (1991).

const float covalent[] = {
    /* LP */ 0.00, /* H  */ 0.23, /* He */ 0.00, /* Li */ 0.68,
    /* Be */ 0.35, /* B  */ 0.83, /* C  */ 0.68, /* N  */ 0.68,
    /* O  */ 0.68, /* F  */ 0.64, /* Ne */ 0.00, /* Na */ 0.97,
    /* Mg */ 1.10, /* Al */ 1.35, /* Si */ 1.20, /* P  */ 1.05,
    /* S  */ 1.02, /* Cl */ 0.99, /* Ar */ 0.00, /* K  */ 1.33,
    /* Ca */ 0.99, /* Sc */ 1.44, /* Ti */ 1.47, /* V  */ 1.33,
    /* Cr */ 1.35, /* Mn */ 1.35, /* Fe */ 1.34, /* Co */ 1.33,
    /* Ni */ 1.50, /* Cu */ 1.52, /* Zn */ 1.45, /* Ga */ 1.22,
    /* Ge */ 1.17, /* As */ 1.21, /* Se */ 1.22, /* Br */ 1.21,
    /* Kr */ 0.00, /* Rb */ 1.47, /* Sr */ 1.12, /* Y  */ 1.78,
    /* Zr */ 1.56, /* Nb */ 1.48, /* Mo */ 1.47, /* Tc */ 1.35,
    /* Ru */ 1.40, /* Rh */ 1.45, /* Pd */ 1.50, /* Ag */ 1.59,
    /* Cd */ 1.69, /* In */ 1.63, /* Sn */ 1.46, /* Sb */ 1.46,
    /* Te */ 1.47, /* I  */ 1.40, /* Xe */ 0.00, /* Cs */ 1.67,
    /* Ba */ 1.34, /* La */ 1.87, /* Ce */ 1.83, /* Pr */ 1.82,
    /* Nd */ 1.81, /* Pm */ 1.80, /* Sm */ 1.80, /* Eu */ 1.99,
    /* Gd */ 1.79, /* Tb */ 1.76, /* Dy */ 1.75, /* Ho */ 1.74,
    /* Er */ 1.73, /* Tm */ 1.72, /* Yb */ 1.94, /* Lu */ 1.72,
    /* Hf */ 1.57, /* Ta */ 1.43, /* W  */ 1.37, /* Re */ 1.35,
    /* Os */ 1.37, /* Ir */ 1.32, /* Pt */ 1.50, /* Au */ 1.50,
    /* Hg */ 1.70, /* Tl */ 1.55, /* Pb */ 1.54, /* Bi */ 1.54,
    /* Po */ 1.68, /* At */ 0.00, /* Rn */ 0.00, /* Fr */ 0.00,
    /* Ra */ 1.90, /* Ac */ 1.88, /* Th */ 1.79, /* Pa */ 1.61,
    /* U  */ 1.58, /* Np */ 1.55, /* Pu */ 1.53, /* Am */ 1.51,
};

const int NUM_COVALENT = sizeof covalent / sizeof covalent[0];

const float standard_mass[] = {
    // from <http://en.wikipedia.org/wiki/List_of_elements_by_atomic_weight>
    /* LP */ 0.0,
    /* H  */ 1.00794,
    /* He */ 4.002602,
    /* Li */ 6.941,
    /* Be */ 9.012182,
    /* B  */ 10.811,
    /* C  */ 12.0107,
    /* N  */ 14.0067,
    /* O  */ 15.9994,
    /* F  */ 18.9984032,
    /* Ne */ 20.1797,
    /* Na */ 22.98976928,
    /* Mg */ 24.305,
    /* Al */ 26.9815386,
    /* Si */ 28.0855,
    /* P  */ 30.973762,
    /* S  */ 32.065,
    /* Cl */ 35.453,
    /* Ar */ 39.948,
    /* K  */ 39.0983,
    /* Ca */ 40.078,
    /* Sc */ 44.955912,
    /* Ti */ 47.867,
    /* V  */ 50.9415,
    /* Cr */ 51.9961,
    /* Mn */ 54.938045,
    /* Fe */ 55.845,
    /* Co */ 58.933195,
    /* Ni */ 58.6934,
    /* Cu */ 63.546,
    /* Zn */ 65.38,
    /* Ga */ 69.723,
    /* Ge */ 72.64,
    /* As */ 74.92160,
    /* Se */ 78.96,
    /* Br */ 79.904,
    /* Kr */ 83.798,
    /* Rb */ 85.4678,
    /* Sr */ 87.62,
    /* Y  */ 88.90585,
    /* Zr */ 91.224,
    /* Nb */ 92.90638,
    /* Mo */ 95.96,
    /* Tc */ 97.9072,
    /* Ru */ 101.07,
    /* Rh */ 102.9055,
    /* Pd */ 106.42,
    /* Ag */ 107.8682,
    /* Cd */ 112.411,
    /* In */ 114.818,
    /* Sn */ 118.71,
    /* Sb */ 121.76,
    /* Te */ 127.60,
    /* I  */ 126.90447,
    /* Xe */ 131.293,
    /* Cs */ 132.9054519,
    /* Ba */ 137.327,
    /* La */ 138.90547,
    /* Ce */ 140.116,
    /* Pr */ 140.90765,
    /* Nd */ 144.242,
    /* Pm */ 145.0,
    /* Sm */ 150.36,
    /* Eu */ 151.964,
    /* Gd */ 157.25,
    /* Tb */ 158.92535,
    /* Dy */ 162.50,
    /* Ho */ 164.93032,
    /* Er */ 167.259,
    /* Tm */ 168.93221,
    /* Yb */ 173.054,
    /* Lu */ 174.9668,
    /* Hf */ 178.49,
    /* Ta */ 180.94788,
    /* W  */ 183.84,
    /* Re */ 186.207,
    /* Os */ 190.23,
    /* Ir */ 192.217,
    /* Pt */ 195.084,
    /* Au */ 196.966469,
    /* Hg */ 200.59,
    /* Tl */ 204.3833,
    /* Pb */ 207.2,
    /* Bi */ 208.98040,
    /* Po */ 208.9824,
    /* At */ 209.9871,
    /* Rn */ 222.0176,
    /* Fr */ 223.0,
    /* Ra */ 226.0,
    /* Ac */ 227.0,
    /* Th */ 232.03806,
    /* Pa */ 231.03588,
    /* U  */ 238.02891,
    /* Np */ 237.0,
    /* Pu */ 244.0,
    /* Am */ 243.0,
    /* Cm */ 247.0,
    /* Bk */ 247.0,
    /* Cf */ 251.0,
    /* Es */ 252.0,
    /* Fm */ 257.0,
    /* Md */ 258.0,
    /* No */ 259.0,
    /* Lr */ 262.0,
    /* Rf */ 261.0,
    /* Db */ 262.0,
    /* Sg */ 266.0,
    /* Bh */ 264.0,
    /* Hs */ 277.0,
    /* Mt */ 268.0,
    /* Ds */ 271.0,
    /* Rg */ 272.0,
    /* Uub */ 285.0,
    /* Uut */ 284.0,
    /* Uuq */ 289.0,
    /* Uup */ 288.0,
    /* Uuh */ 292.0,
    /* Uus */ 291.0,
    /* Uuo */ 293.0,
};

const int NUM_MASS = sizeof standard_mass / sizeof standard_mass[0];

const char *
Element::name() const {
    if (as >= NUM_SYMBOLS)
        return "??";
    return symbols[as];
}

float
Element::mass() const {
    if (as >= NUM_MASS)
        return 0.00;
    return standard_mass[as];
}

Element::AS
Element::atomic_number(const char *name)
{
    char const * const    *e;
    char            symbol[2];

    if (name == NULL)
        return LONE_PAIR;
    
    if (isdigit(name[0]) || name[0] == ' ') {
        if (islower(name[1]))
            symbol[0] = toupper(name[1]);
        else
            symbol[0] = name[1];
        symbol[1] = '\0';
    } else {
        if (islower(name[0]))
            symbol[0] = toupper(name[0]);
        else
            symbol[0] = name[0];
        if (isupper(name[1]))
            symbol[1] = tolower(name[1]);
        else
            symbol[1] = name[1];
    }

    if (symbol[1] == '\0')
        switch (symbol[0]) {
        case 'H': return H;
        case 'D': return D;    // deuterium
        case 'T': return T;    // tritium
        case 'B': return B;
        case 'C': return C;
        case 'N': return N;
        case 'O': return O;
        case 'F': return F;
        case 'P': return P;
        case 'S': return S;
        case 'K': return K;
        case 'V': return V;
        case 'Y': return Y;
        case 'I': return I;
        case 'W': return W;
        case 'U': return U;
        default: return LONE_PAIR;
        }

    for (e = symbols + 1; e < &symbols[Uub + 1]; e += 1)
        if (symbol[0] == (*e)[0] && symbol[1] == (*e)[1])
            return AS(e - symbols);
    return LONE_PAIR;
}

float
Element::bond_radius(Element a)
{
    if (a.number() < 0 || a.number() >= NUM_COVALENT)
        return 0.0;
    else
        return covalent[a.number()];
}

float
Element::bond_length(Element a0, Element a1)
{
    return bond_radius(a0) + bond_radius(a1);
}

bool
Element::is_metal() const
{
    int n = number();
    if ((n >= 5 && n <= 10) || n <= 2 || (n >= 14 && n <= 18))
        return false;    // Check most common elements first.
    return ((n >= 11 && n <= 13) || (n >= 19 && n <= 32) ||
        (n >= 3 && n <= 4) || (n >= 37 && n <= 51) ||
        (n >= 55 && n <= 84) || (n >= 87 && n <= 103));
}

}  // namespace atomstruct
