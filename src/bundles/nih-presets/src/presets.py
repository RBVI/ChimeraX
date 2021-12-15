# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

base_setup = [
    "windowsize 512 512",
    "graphics bgcolor white",
    "color name marine 0,50,100",
    "color name forest 13.3,54.5,13.3",
    "color name tangerine 95.3,51.8,0",
    "color name grape 64.3,0,86.7",
    "color name nih_blue 12.5,33.3,54.1",
    "color name jmol_carbon 56.5,56.5,56.5",
    "color name bond_purple 57.6,43.9,85.9",
    "color name struts_grey 48,48,48",
    "color name carbon_grey 22.2,22.2,22.2"
]
base_macro_model = [
    "delete solvent",
    "delete H"
]

def run_preset(session, name, mgr):
    if name == "ribbon":
        cmd = base_setup + base_macro_model + [
            # 'struts' doesn't work right until ribbon data structures get updated, so wait 1 frame
            "preset 'initial styles' cartoon; wait 1",
            "nucleotides ladder radius 1.2",
            "color white",
            "color helix marine; color strand firebrick; color coil goldenrod; color nucleic-acid forest",
            "color :A:C:G:U grape",
            "color byatom",
            "select (C & ligand) | (C & ligand :< 5 & ~nucleic-acid) | (C & protein) | (C & disulfide)",
            "color sel carbon_grey atoms",
            "color ligand | protein & sideonly byhet atoms",
            "~select"
        ]
    else:
        from chimerax.core.errors import UserError
        raise UserError("Unknown NIH3D preset '%s'" % name)
    cmd = "; ".join(cmd)
    mgr.execute(cmd)
