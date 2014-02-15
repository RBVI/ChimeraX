// vim: set expandtab ts=4 sw=4:
#include "residues.h"
#include "resinternal.h"

#include "resDescrip.h"

extern void    restmpl_init_amino(ResInitMap *);
extern void    restmpl_init_camino(ResInitMap *);
extern void    restmpl_init_namino(ResInitMap *);
extern void    restmpl_init_nucleic(ResInitMap *);
extern void    restmpl_init_general(ResInitMap *);
extern void    restmpl_init_ions(ResInitMap *);

static TmplMolecule    *start_mol = NULL, *middle_mol, *end_mol;
static ResInitMap    resmap;

void
restmpl_init()
{
    start_mol = new TmplMolecule;
    middle_mol = new TmplMolecule;
    end_mol = new TmplMolecule;
    restmpl_init_amino(&resmap);
    restmpl_init_camino(&resmap);
    restmpl_init_namino(&resmap);
    restmpl_init_nucleic(&resmap);
    restmpl_init_general(&resmap);
    restmpl_init_ions(&resmap);
    TmplMolecule *mols[3] = { start_mol, middle_mol, end_mol };
    for (unsigned int i = 0; i != sizeof(res_descripts)/sizeof(ResDescript);
                                    ++i) {
        for (unsigned int mi = 0; mi != 3; ++mi) {
            TmplMolecule *m = mols[mi];
            TmplResidue *r = m->find_residue(std::string(res_descripts[i].name));
            if (r == NULL)
                continue;
            r->description(std::string(res_descripts[i].descrip));
        }
    }
}

// need function to find template for residue and fill in what's missing
const TmplResidue *
find_template_residue(const std::string &name, bool start, bool end)
{
    bool new_r = false;
    std::string mapped_name = name;

    if (start_mol == NULL)
        restmpl_init();

    // assume that most users will want HID (who knows?)
    if (mapped_name == "HIS")
        mapped_name.assign("HID");

    // DNA templates have old names, remap...
    if (mapped_name == "DA")
        mapped_name.assign("A");
    if (mapped_name == "DC")
        mapped_name.assign("C");
    if (mapped_name == "DG")
        mapped_name.assign("G");
    if (mapped_name == "DT")
        mapped_name.assign("T");

    ResInitMap::iterator i = resmap.find(mapped_name);
    ResInit *ri = (i == resmap.end()) ? NULL : &i->second;
    TmplResidue *r;
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
            r->template_assign(&TmplAtom::set_idatm_type,
                    "idatm", "templates", "idatmres");
        } catch (TA_NoTemplate) {
            // ssshhhh!
        }
    }
    return r;
}
