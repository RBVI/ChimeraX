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

#define ATOMSTRUCT_EXPORT
#include "residues.h"
#include "resinternal.h"

#include "resDescrip.h"

namespace tmpl {

extern void    restmpl_init_amino(ResInitMap *);
extern void    restmpl_init_camino(ResInitMap *);
extern void    restmpl_init_namino(ResInitMap *);
extern void    restmpl_init_nucleic(ResInitMap *);
extern void    restmpl_init_general(ResInitMap *);
extern void    restmpl_init_ions(ResInitMap *);

static Molecule    *start_mol = NULL, *middle_mol, *end_mol;
static ResInitMap    resmap;

void
restmpl_init()
{
    start_mol = new Molecule;
    middle_mol = new Molecule;
    end_mol = new Molecule;
    restmpl_init_amino(&resmap);
    restmpl_init_camino(&resmap);
    restmpl_init_namino(&resmap);
    restmpl_init_nucleic(&resmap);
    restmpl_init_general(&resmap);
    restmpl_init_ions(&resmap);
    Molecule *mols[3] = { start_mol, middle_mol, end_mol };
    for (unsigned int i = 0; i != sizeof(res_descripts)/sizeof(ResDescript);
                                    ++i) {
        for (unsigned int mi = 0; mi != 3; ++mi) {
            Molecule *m = mols[mi];
            Residue *r = m->find_residue(res_descripts[i].name);
            if (r == NULL)
                continue;
            r->description(std::string(res_descripts[i].descrip));
        }
    }
}

// need function to find template for residue and fill in what's missing
const Residue *
find_template_residue(const ResName& name, bool start, bool end)
{
    bool new_r = false;
    ResName mapped_name = name;

    if (start_mol == NULL)
        restmpl_init();

    // since which HIS template to use is guesswork anyway, use HIP
    // since that provides the connectivity of both hydrogens
    if (mapped_name == "HIS")
        mapped_name = "HIP";

    ResInitMap::iterator i = resmap.find(mapped_name);
    ResInit *ri = (i == resmap.end()) ? NULL : &i->second;
    Residue *r;
    if (start) {
        r = start_mol->find_residue(mapped_name);
        if (r == NULL && ri && ri->start) {
            new_r = true;
            r = ri->start(start_mol);
        }
    }
    else if (end) {
        r = end_mol->find_residue(mapped_name);
        if (r == NULL && ri && ri->end) {
            new_r = true;
            r = ri->end(end_mol);
        }
    } else
        r = NULL;
    if (r == NULL) {
        r = middle_mol->find_residue(mapped_name);
        if (r == NULL && ri && ri->middle) {
            new_r = true;
            r = ri->middle(middle_mol);
        }
    }
    if (new_r) {
        try {
            r->template_assign(&Atom::set_idatm_type,
                    "idatm", "templates", "idatmres");
        } catch (TA_NoTemplate) {
            // ssshhhh!
        }
    }
    return r;
}

}  // namespace tmpl
