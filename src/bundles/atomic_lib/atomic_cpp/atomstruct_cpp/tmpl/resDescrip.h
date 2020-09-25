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

#ifndef templates_resDescrip
#define templates_resDescrip

#include "../imex.h"
#include "../string_types.h"

namespace tmpl {

typedef struct resDescrip {
    ResName  name;
    const char  *descrip;
} ResDescript;

ATOMSTRUCT_IMEX ResDescript res_descripts[] = {
    { "ALA", "alanine" },
    { "ARG", "arginine" },
    { "ASH", "protonated aspartic acid" },
    { "ASN", "asparagine" },
    { "ASP", "aspartic acid" },
    { "CYS", "cysteine" },
    { "CYX", "cysteine with disulfide bond" },
    { "GLH", "protonated glutamic acid" },
    { "GLN", "glutamine" },
    { "GLU", "glutamic acid" },
    { "GLY", "glycine" },
    { "HIS", "histidine" },
    { "HID", "delta-protonated histidine" },
    { "HIE", "epsilon-protonated histidine" },
    { "HIP", "double-protonated histidine" },
    { "HYP", "hydroxyproline" },
    { "ILE", "isoleucine" },
    { "LEU", "leucine" },
    { "LYS", "lysine" },
    { "MET", "methionine" },
    { "PHE", "phenylalanine" },
    { "PRO", "proline" },
    { "SER", "serine" },
    { "THR", "threonine" },
    { "TRP", "tryptophan" },
    { "TYR", "tyrosine" },
    { "VAL", "valine" }
};

}  // namespace tmpl

#endif  // templates_resDescrip
