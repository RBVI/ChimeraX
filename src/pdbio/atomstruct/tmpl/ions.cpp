// vim: set expandtab ts=4 sw=4:
#include "resinternal.h"

namespace tmpl {

using atomstruct::Element;

static Residue *
init_Cs(Molecule *m)
{
    Coord c;
    Residue *r = m->new_residue("Cs+");
    Atom *atom_Csp = m->new_atom("Cs+", Element(55));
    r->add_atom(atom_Csp);
    c.set_xyz(0.0,0.0,0.0);
    atom_Csp->set_coord(c);
    return r;
}

static Residue *
init_K(Molecule *m)
{
    Coord c;
    Residue *r = m->new_residue("K+");
    Atom *atom_Kp = m->new_atom("K+", Element(19));
    r->add_atom(atom_Kp);
    c.set_xyz(0.0,0.0,0.0);
    atom_Kp->set_coord(c);
    return r;
}

static Residue *
init_Li(Molecule *m)
{
    Coord c;
    Residue *r = m->new_residue("Li+");
    Atom *atom_Lip = m->new_atom("Li+", Element(3));
    r->add_atom(atom_Lip);
    c.set_xyz(0.0,0.0,0.0);
    atom_Lip->set_coord(c);
    return r;
}

static Residue *
init_Na(Molecule *m)
{
    Coord c;
    Residue *r = m->new_residue("Na+");
    Atom *atom_Nap = m->new_atom("Na+", Element(11));
    r->add_atom(atom_Nap);
    c.set_xyz(0.0,0.0,0.0);
    atom_Nap->set_coord(c);
    return r;
}

static Residue *
init_Rb(Molecule *m)
{
    Coord c;
    Residue *r = m->new_residue("Rb+");
    Atom *atom_Rbp = m->new_atom("Rb+", Element(37));
    r->add_atom(atom_Rbp);
    c.set_xyz(0.0,0.0,0.0);
    atom_Rbp->set_coord(c);
    return r;
}

void
restmpl_init_ions(ResInitMap *rim)
{
    (*rim)[std::string("Cs+")].middle = init_Cs;
    (*rim)[std::string("K+")].middle = init_K;
    (*rim)[std::string("Li+")].middle = init_Li;
    (*rim)[std::string("Na+")].middle = init_Na;
    (*rim)[std::string("Rb+")].middle = init_Rb;
}

}  // namespace tmpl
