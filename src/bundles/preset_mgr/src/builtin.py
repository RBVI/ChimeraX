# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

nospheres = (["style (protein|nucleic|solvent) & @@draw_mode=0 stick"])
cardef    = (["show nucleic", "hide protein|solvent|H", "surf hide"] +
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

def run_preset(session, name, mgr):
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
    elif name == "publication 1 (silhouettes)":
        cmd = ["set bg white; graphics silhouettes t; lighting depthCue f"]
    elif name == "publication 2 (depth-cued)":
        cmd = ["set bg white; graphics silhouettes f; lighting depthCue t"]
    elif name == "interactive":
        cmd = ["~set bg; graphics silhouettes f; lighting depthCue t"]
    cmd = "; ".join(cmd)
    mgr.execute(cmd)
