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

nospheres = (["style (protein|nucleic|solvent) & @@draw_mode=0 stick"])
cardef    = (["surf hide"] +
             nospheres +
             ["cartoon",
              "cartoon style modeh def arrows t arrowshelix f "
                "arrowscale 2 wid 2 thick 0.4 sides 12 div 20",
              "cartoon style ~(nucleic|strand) x round",
              "cartoon style (nucleic|strand) x rect"])
cylinders = (cardef +
             ["cartoon style protein modeh tube rad 2 sides 24 thick 0.6"])
licorice  = (cardef +
             ["cartoon style protein modeh default arrows f x round width 1 thick 1"])

def run_preset(session, name):
    if name == "ribbons/slabs":
        cmd = cardef + ["nucleotides tube/slab shape box"]
    elif name == "cylinders/stubs":
        cmd = cylinders + ["cartoon style nucleic x round "
                           "width 1.6 thick 1.6; nucleotides stubs"]
    elif name == "licorice/ovals":
        cmd = licorice + ["cartoon style nucleic x round width 1.6 thick 1.6",
                          "nucleotides tube/slab shape ellipsoid"]
    elif name == "ghostly white":
        cmd = nospheres + ["surface; color white targ s trans 80"]
    elif name == "atomic coloring (transparent)":
        cmd = nospheres + ["surface; color fromatoms targ s trans 70"]
    elif name == "chain ID coloring (opaque)":
        cmd = nospheres + ["surface; color bychain targ s trans 0"]
    elif name == "publication":
        cmd = ["set bg white; set silhouettes t"]
    elif name == "interactive":
        cmd = ["~set bg; set silhouettes f"]
    cmd = "; ".join(cmd)
    from chimerax.core.commands import run
    run(session, cmd, log=False)
    session.logger.info("Preset expands to these ChimeraX commands: "
                        "<i>%s</i>" % cmd, is_html=True)
