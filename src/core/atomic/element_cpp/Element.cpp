// vi: set expandtab ts=4 sw=4:

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

#include <set>
#include <ctype.h>

#define ELEMENT_EXPORT
#include "Element.h"

namespace element {

#if 0
template class ELEMENT_IMEX std::map<int, const Element*>;
template class ELEMENT_IMEX std::set<std::string>;
template class ELEMENT_IMEX std::set<int>;
#endif

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
    "Hs", "Mt", "Ds", "Rg", "Cn", "Uut", "Fl", "Uup", "Uuh",
    "Uus", "Uuo"
};

const int NUM_SYMBOLS = sizeof symbols / sizeof symbols[0];

//    The following covalent bond radii are taken from: Elaine C. Meng
//    and Richard A. Lewis, "Determination of Molecular Topology and
//    Atomic Hybridization States from Heavy Atom Coordinates," Journal
//    of Computational Chemistry, Vol. 12, No. 7, pp. 891-898 (1991).

const float covalent[] = {
    /* LP */ 0.00f, /* H  */ 0.23f, /* He */ 0.00f, /* Li */ 0.68f,
    /* Be */ 0.35f, /* B  */ 0.83f, /* C  */ 0.68f, /* N  */ 0.68f,
    /* O  */ 0.68f, /* F  */ 0.64f, /* Ne */ 0.00f, /* Na */ 0.97f,
    /* Mg */ 1.10f, /* Al */ 1.35f, /* Si */ 1.20f, /* P  */ 1.05f,
    /* S  */ 1.02f, /* Cl */ 0.99f, /* Ar */ 0.00f, /* K  */ 1.33f,
    /* Ca */ 0.99f, /* Sc */ 1.44f, /* Ti */ 1.47f, /* V  */ 1.33f,
    /* Cr */ 1.35f, /* Mn */ 1.35f, /* Fe */ 1.34f, /* Co */ 1.33f,
    /* Ni */ 1.50f, /* Cu */ 1.52f, /* Zn */ 1.45f, /* Ga */ 1.22f,
    /* Ge */ 1.17f, /* As */ 1.21f, /* Se */ 1.22f, /* Br */ 1.21f,
    /* Kr */ 0.00f, /* Rb */ 1.47f, /* Sr */ 1.12f, /* Y  */ 1.78f,
    /* Zr */ 1.56f, /* Nb */ 1.48f, /* Mo */ 1.47f, /* Tc */ 1.35f,
    /* Ru */ 1.40f, /* Rh */ 1.45f, /* Pd */ 1.50f, /* Ag */ 1.59f,
    /* Cd */ 1.69f, /* In */ 1.63f, /* Sn */ 1.46f, /* Sb */ 1.46f,
    /* Te */ 1.47f, /* I  */ 1.40f, /* Xe */ 0.00f, /* Cs */ 1.67f,
    /* Ba */ 1.34f, /* La */ 1.87f, /* Ce */ 1.83f, /* Pr */ 1.82f,
    /* Nd */ 1.81f, /* Pm */ 1.80f, /* Sm */ 1.80f, /* Eu */ 1.99f,
    /* Gd */ 1.79f, /* Tb */ 1.76f, /* Dy */ 1.75f, /* Ho */ 1.74f,
    /* Er */ 1.73f, /* Tm */ 1.72f, /* Yb */ 1.94f, /* Lu */ 1.72f,
    /* Hf */ 1.57f, /* Ta */ 1.43f, /* W  */ 1.37f, /* Re */ 1.35f,
    /* Os */ 1.37f, /* Ir */ 1.32f, /* Pt */ 1.50f, /* Au */ 1.50f,
    /* Hg */ 1.70f, /* Tl */ 1.55f, /* Pb */ 1.54f, /* Bi */ 1.54f,
    /* Po */ 1.68f, /* At */ 0.00f, /* Rn */ 0.00f, /* Fr */ 0.00f,
    /* Ra */ 1.90f, /* Ac */ 1.88f, /* Th */ 1.79f, /* Pa */ 1.61f,
    /* U  */ 1.58f, /* Np */ 1.55f, /* Pu */ 1.53f, /* Am */ 1.51f,
};

const int NUM_COVALENT = sizeof covalent / sizeof covalent[0];

const float standard_mass[] = {
    // from <http://en.wikipedia.org/wiki/List_of_elements_by_atomic_weight>
    /* LP */ 0.0f,
    /* H  */ 1.00794f,
    /* He */ 4.002602f,
    /* Li */ 6.941f,
    /* Be */ 9.012182f,
    /* B  */ 10.811f,
    /* C  */ 12.0107f,
    /* N  */ 14.0067f,
    /* O  */ 15.9994f,
    /* F  */ 18.9984032f,
    /* Ne */ 20.1797f,
    /* Na */ 22.98976928f,
    /* Mg */ 24.305f,
    /* Al */ 26.9815386f,
    /* Si */ 28.0855f,
    /* P  */ 30.973762f,
    /* S  */ 32.065f,
    /* Cl */ 35.453f,
    /* Ar */ 39.948f,
    /* K  */ 39.0983f,
    /* Ca */ 40.078f,
    /* Sc */ 44.955912f,
    /* Ti */ 47.867f,
    /* V  */ 50.9415f,
    /* Cr */ 51.9961f,
    /* Mn */ 54.938045f,
    /* Fe */ 55.845f,
    /* Co */ 58.933195f,
    /* Ni */ 58.6934f,
    /* Cu */ 63.546f,
    /* Zn */ 65.38f,
    /* Ga */ 69.723f,
    /* Ge */ 72.64f,
    /* As */ 74.92160f,
    /* Se */ 78.96f,
    /* Br */ 79.904f,
    /* Kr */ 83.798f,
    /* Rb */ 85.4678f,
    /* Sr */ 87.62f,
    /* Y  */ 88.90585f,
    /* Zr */ 91.224f,
    /* Nb */ 92.90638f,
    /* Mo */ 95.96f,
    /* Tc */ 97.9072f,
    /* Ru */ 101.07f,
    /* Rh */ 102.9055f,
    /* Pd */ 106.42f,
    /* Ag */ 107.8682f,
    /* Cd */ 112.411f,
    /* In */ 114.818f,
    /* Sn */ 118.71f,
    /* Sb */ 121.76f,
    /* Te */ 127.60f,
    /* I  */ 126.90447f,
    /* Xe */ 131.293f,
    /* Cs */ 132.9054519f,
    /* Ba */ 137.327f,
    /* La */ 138.90547f,
    /* Ce */ 140.116f,
    /* Pr */ 140.90765f,
    /* Nd */ 144.242f,
    /* Pm */ 145.0f,
    /* Sm */ 150.36f,
    /* Eu */ 151.964f,
    /* Gd */ 157.25f,
    /* Tb */ 158.92535f,
    /* Dy */ 162.50f,
    /* Ho */ 164.93032f,
    /* Er */ 167.259f,
    /* Tm */ 168.93221f,
    /* Yb */ 173.054f,
    /* Lu */ 174.9668f,
    /* Hf */ 178.49f,
    /* Ta */ 180.94788f,
    /* W  */ 183.84f,
    /* Re */ 186.207f,
    /* Os */ 190.23f,
    /* Ir */ 192.217f,
    /* Pt */ 195.084f,
    /* Au */ 196.966469f,
    /* Hg */ 200.59f,
    /* Tl */ 204.3833f,
    /* Pb */ 207.2f,
    /* Bi */ 208.98040f,
    /* Po */ 208.9824f,
    /* At */ 209.9871f,
    /* Rn */ 222.0176f,
    /* Fr */ 223.0f,
    /* Ra */ 226.0f,
    /* Ac */ 227.0f,
    /* Th */ 232.03806f,
    /* Pa */ 231.03588f,
    /* U  */ 238.02891f,
    /* Np */ 237.0f,
    /* Pu */ 244.0f,
    /* Am */ 243.0f,
    /* Cm */ 247.0f,
    /* Bk */ 247.0f,
    /* Cf */ 251.0f,
    /* Es */ 252.0f,
    /* Fm */ 257.0f,
    /* Md */ 258.0f,
    /* No */ 259.0f,
    /* Lr */ 262.0f,
    /* Rf */ 261.0f,
    /* Db */ 262.0f,
    /* Sg */ 266.0f,
    /* Bh */ 264.0f,
    /* Hs */ 277.0f,
    /* Mt */ 268.0f,
    /* Ds */ 271.0f,
    /* Rg */ 272.0f,
    /* Cn */ 285.0f,
    /* Uut */ 284.0f,
    /* Fl */ 289.0f,
    /* Uup */ 288.0f,
    /* Uuh */ 292.0f,
    /* Uus */ 291.0f,
    /* Uuo */ 293.0f,
};

const int NUM_MASS = sizeof standard_mass / sizeof standard_mass[0];

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

    for (e = symbols + 1; e < symbols + NUM_SYMBOLS; e += 1)
        if (symbol[0] == (*e)[0] && symbol[1] == (*e)[1])
            return AS(e - symbols);
    return LONE_PAIR;
}

float
Element::bond_radius(const Element& e)
{
    if (e.number() < 0 || e.number() >= NUM_COVALENT)
        return 0.0;
    else
        return covalent[e.number()];
}

float
Element::bond_length(const Element& e0, const Element& e1)
{
    return bond_radius(e0) + bond_radius(e1);
}

std::map<int, const Element*>  Element::_elements;

const Element&
Element::get_element(int i)
{
    auto ei = _elements.find(i);
    if (ei != _elements.end())
        return *(*ei).second;
    Element* e = new Element((AS)i);
    _elements[i] = e;
    return *e;
}

std::set<int> Element::_alkali_metals = {
    3 /* Li */, 11 /* Na */, 19 /* K */, 37 /* Rb */, 55 /* Cs */, 87 /* Fr */
};

std::set<int> Element::_halogens = {
    9 /* F */, 17 /* Cl */, 35 /* Br */, 53 /* I */, 85 /* At */, 117 /* Uus */
};

std::set<int> Element::_noble_gases = {
    2 /* He */, 10 /* Ne */, 18 /* Ar */, 36 /* Kr */, 54 /* Xe */, 86 /* Rn */, 118 /* UUo */
};

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

const char *
Element::name() const {
    if (as >= NUM_SYMBOLS)
        return "??";
    return symbols[as];
}

std::set<std::string>  Element::_names;

const std::set<std::string>&
Element::names()
{
    if (_names.empty()) {
        // insert all real (i.e. non-LP) element names
        for (int i = 1; i < NUM_SYMBOLS; ++i)
            _names.insert(symbols[i]);
    }
    return _names;   
}

float
Element::mass() const {
    if (as >= NUM_MASS)
        return 0.00;
    return standard_mass[as];
}

int
Element::valence() const
{
    int n = number();
    if (n == 1) return 1;
    if (n <= 20) return (n - 2) % 8;
    if (n <= 56) return (n - 18) % 18;
    return (n - 54) % 32;
}

}  // namespace element
