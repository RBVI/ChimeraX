// vim: set expandtab ts=4 sw=4:
#include "resinternal.h"


static TmplResidue *
init_Cs(TmplMolecule *m)
{
    TmplCoord c;
    TmplResidue *r = m->new_residue("Cs+");
    TmplAtom *atom_Csp = m->new_atom("Cs+", Element(55));
    r->add_atom(atom_Csp);
    c.set_xyz(0.0,0.0,0.0);
    atom_Csp->set_coord(c);
    return r;
}

static TmplResidue *
init_K(TmplMolecule *m)
{
    TmplCoord c;
    TmplResidue *r = m->new_residue("K+");
    TmplAtom *atom_Kp = m->new_atom("K+", Element(19));
    r->add_atom(atom_Kp);
    c.set_xyz(0.0,0.0,0.0);
    atom_Kp->set_coord(c);
    return r;
}

static TmplResidue *
init_Li(TmplMolecule *m)
{
    TmplCoord c;
    TmplResidue *r = m->new_residue("Li+");
    TmplAtom *atom_Lip = m->new_atom("Li+", Element(3));
    r->add_atom(atom_Lip);
    c.set_xyz(0.0,0.0,0.0);
    atom_Lip->set_coord(c);
    return r;
}

static TmplResidue *
init_Na(TmplMolecule *m)
{
    TmplCoord c;
    TmplResidue *r = m->new_residue("Na+");
    TmplAtom *atom_Nap = m->new_atom("Na+", Element(11));
    r->add_atom(atom_Nap);
    c.set_xyz(0.0,0.0,0.0);
    atom_Nap->set_coord(c);
    return r;
}

static TmplResidue *
init_Rb(TmplMolecule *m)
{
    TmplCoord c;
    TmplResidue *r = m->new_residue("Rb+");
    TmplAtom *atom_Rbp = m->new_atom("Rb+", Element(37));
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
