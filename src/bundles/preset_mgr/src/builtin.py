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

def register_builtin_presets(session):
    cartoon1 = "surf hide; car style modeh def arrows t arrowshelix f arrowscale 2 wid 2 thick 0.4 sides 12" \
        " div 20; car style ~(nucleic|strand) x round; car style (nucleic|strand) x rect"
    cylinders = "%s; ; car style protein modeh tube rad 2 sides 24 thick 0.6" % cartoon1
    licorice = "%s; car style protein modeh default arrows f x oval width 1 thick 1" % cartoon1
    preset_info = {
        "Cartoon/Nucleotides": [
            ("ribbons/slabs",
                "surf hide; car style modeh def arrows t arrowshelix f arrowscale 2 wid 2 thick 0.4 sides 12"
                " div 20; car style ~(nucleic|strand) x round; car style (nucleic|strand) x rect;"
                " nuc tube/slab shape box"),
            ("cylinders/stubs",
                "surf hide; cartoon; %s; car style nucleic x oval width 1.6 thick 1.6; nuc stubs" % cylinders),
            ("licorice/ovals",
                "surf hide; cartoon; %s; car style nucleic x oval width 1.6 thick 1.6; nuc tube/slab"
                " shape ellipsoid" % licorice),
        ],
        "Surface": [
            ("ghostly white", "surface; color white targ s; trans 80 targ s"),
            ("color by model", "surface; color bymodel targ s; trans 70 targ s"),
            ("color by atom", "surface; color fromatoms targ s; trans 70 targ s"),
        ],
        "Overall Look": [
            ("publication", "set bg white; set silhouettes t"),
            ("interactive", "~set bg; set silhouettes f"),
        ],
    }
    from chimerax.core.commands import run
    for cat_name, cat_presets in preset_info.items():
        session.presets.add_presets(cat_name,
            [(name, lambda run=run, ses=session, cmd=cmd: run(ses, cmd, log=False))
                for name, cmd in cat_presets])

