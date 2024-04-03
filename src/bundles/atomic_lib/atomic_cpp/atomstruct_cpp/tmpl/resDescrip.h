// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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
