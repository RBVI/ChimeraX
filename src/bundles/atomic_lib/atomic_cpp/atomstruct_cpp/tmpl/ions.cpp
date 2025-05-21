// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This particular file is part of the ChimeraX library. You can also
 * redistribute and/or modify it under the terms of the GNU Lesser General
 * Public License version 2.1 as published by the Free Software Foundation.
 * For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
 * LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
 * VERSION 2.1
 *
 * This notice must be embedded in or attached to all copies, including partial
 * copies, of the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#define ATOMSTRUCT_EXPORT
#include "Atom.h"
#include <element/Element.h>
#include "Molecule.h"
#include "Residue.h"
#include "resinternal.h"

namespace tmpl {

using element::Element;


static Residue *
init_Cs(Molecule *m)
{
	Coord c;
	Residue *r = m->new_residue("Cs+");
	Atom *atom_Csp = m->new_atom("Cs+", Element::get_element(55));
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
	Atom *atom_Kp = m->new_atom("K+", Element::get_element(19));
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
	Atom *atom_Lip = m->new_atom("Li+", Element::get_element(3));
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
	Atom *atom_Nap = m->new_atom("Na+", Element::get_element(11));
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
	Atom *atom_Rbp = m->new_atom("Rb+", Element::get_element(37));
	r->add_atom(atom_Rbp);
	c.set_xyz(0.0,0.0,0.0);
	atom_Rbp->set_coord(c);
	return r;
}

void
restmpl_init_ions(ResInitMap *rim)
{
	(*rim)["Cs+"].middle = init_Cs;
	(*rim)["K+"].middle = init_K;
	(*rim)["Li+"].middle = init_Li;
	(*rim)["Na+"].middle = init_Na;
	(*rim)["Rb+"].middle = init_Rb;
}
}  // namespace tmpl
