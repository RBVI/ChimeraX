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
init_TIP3(Molecule *m)
{
	Coord c;
	Residue *r = m->new_residue("WAT");
	Atom *atom_H2 = m->new_atom("H2", Element::get_element(1));
	r->add_atom(atom_H2);
	c.set_xyz(0.9572000000,0.0000000000,0.0000000000);
	atom_H2->set_coord(c);
	Atom *atom_O = m->new_atom("O", Element::get_element(8));
	r->add_atom(atom_O);
	c.set_xyz(-0.2399879329,0.9266270189,0.0000000000);
	atom_O->set_coord(c);
	Atom *atom_H1 = m->new_atom("H1", Element::get_element(1));
	r->add_atom(atom_H1);
	c.set_xyz(0.0000000000,0.0000000000,0.0000000000);
	atom_H1->set_coord(c);
	(void) m->new_bond(atom_H2, atom_O);
	(void) m->new_bond(atom_O, atom_H1);
	return r;
}

void
restmpl_init_general(ResInitMap *rim)
{
	(*rim)["WAT"].middle = init_TIP3;
}
}  // namespace tmpl
