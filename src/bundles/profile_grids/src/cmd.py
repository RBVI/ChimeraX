# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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

from chimerax.core.commands.cli import RegisteredCommandInfo

registry = RegisteredCommandInfo()

# all the commands use the trick that the run() function
# temporarily puts a copy of the Profile Grid instance into
# the global namespace as '_pg'

def label_cmd(session, residues=None, *, height=1.5, no_data_color=(180,180,180,255), offset=(0,0,3),
        on_top=False, palette=None, range_=None):
    from chimerax.core.errors import UserError
    alignment = _pg.alignment
    assoc_residues = alignment.associated_residues()
    if not assoc_residues:
        raise UserError("No chains are associated with the alignment")
    if residues is None:
        label_residues = assoc_residues
    else:
        assoc_set = set(assoc_residues)
        label_residues = [r for r in residues if r in assoc_set]
        if not label_residues:
            raise UserError("None of the specified residues are associated with the alignment")

    if palette is None:
        palette, *ignore = ColormapArg.parse("0,white:1,blue", session)

    res_to_col = {}
    for mms in alignment.match_maps.values():
        for mm in mms.values():
            res_to_col.update(mm.res_to_pos)
    from chimerax.surface.colorvol import _use_full_range, _colormap_with_range
    numeric_range = (0.0, 1.0) if _use_full_range(range_, palette) else range_
    colormap = _colormap_with_range(palette, numeric_range)
    from chimerax.label.label3d import labels_model, ResidueLabel
    lms = set()
    full_rows, columns = _pg.grid_canvas.grid_data.shape
    row_labels = _pg.grid_canvas.existing_row_labels
    divisor = sum(_pg.grid_canvas.weights)
    for r in label_residues:
        lm = labels_model(r.structure, create=True)
        settings = {'height': height, 'offset': offset}
        lm.add_labels([r], ResidueLabel, session.main_view, settings, on_top)
        displayed_row = 0
        cell_data = []
        for i in range(full_rows):
            if i in _pg.grid_canvas.empty_rows:
                continue
            val = _pg.grid_canvas.grid_data[i,res_to_col[r]]
            fraction = val / divisor
            cell_data.append((row_labels[displayed_row], colormap.interpolated_rgba8([fraction])[0]))
            displayed_row += 1

        rlabels = lm.labels([r])
        rlabels[0].custom_image = _label_rgba(r, cell_data, no_data_color)
        lms.add(lm)
    for lm in lms:
        lm.update_labels()

def _label_rgba(res, cell_data, no_data_color):
    from Qt.QtGui import QImage, QPainter, QFont, QColor, QBrush, QPen, QFontMetrics
    wc, hc = 40, 40  # Cell size in pixels
    font_size = 40
    xpad, ypad = 5, 5  # Font offset pixels
    rows = cols = 0
    while rows * cols < len(cell_data):
        cols += 1
        if rows * cols >= len(cell_data):
            break
        rows += 1
    w, h = cols * wc, (rows+1) * hc
    font = "Helvetica"
    p = QPainter()
    ti = QImage(w, h, QImage.Format.Format_ARGB32)
    p.begin(ti)
    p.setCompositionMode(p.CompositionMode_Source)
    from Qt.QtCore import Qt
    pbr = QBrush(Qt.SolidPattern)
    p.setBrush(pbr)
    ppen = QPen(Qt.NoPen)
    p.setPen(ppen)

    # Title
    from chimerax.core.colors import contrast_with
    pbr.setColor(QColor(*no_data_color))
    p.fillRect(0, 0, w, hc, pbr)
    f = QFont(font)
    f.setPixelSize(font_size)
    fm = QFontMetrics(f)
    small_f = QFont(font)
    small_f.setPixelSize(round(font_size/2))
    small_xpad, small_ypad = xpad/2, ypad/2
    small_fm = QFontMetrics(small_f)
    p.setFont(f)
    p.setPen(QColor(*[round(c*255) for c in contrast_with(no_data_color)]))
    p.drawText(wc+xpad, hc-ypad, f"{res.one_letter_code}{res.number}")

    # Grid cells
    for r in range(rows):
        for c in range(cols):
            # background
            x, y = c * wc, (r+1) * hc
            try:
                text, color = cell_data[r * cols + c]
            except IndexError:
                text = None
                color = no_data_color
            pbr.setColor(QColor(*tuple(color)))
            p.setPen(ppen)
            p.fillRect(x, y, wc, hc, pbr)
            if text is None:
                continue
            # text
            p.setPen(QColor(*[round(c*255) for c in contrast_with([c/255 for c in color])]))
            if len(text) == 1:
                metrics = fm
                p.setFont(f)
                xp, yp = xpad, ypad
            else:
                metrics = small_fm
                p.setFont(small_f)
                xp, yp = small_xpad, small_ypad
            bx, by, bw, bh = metrics.boundingRect(text).getRect()
            # a lot of guesswork here
            tx = round(x + (wc - bx/2 - bw - xp)/2)
            ty = round(y - (hc - by/2 - bh - yp)/2 + hc)
            p.drawText(tx, ty, text)

    # Convert to numpy rgba array
    from chimerax.graphics import qimage_to_numpy
    rgba = qimage_to_numpy(ti)
    p.end()
    return rgba


from chimerax.core.commands import CmdDesc, register
from chimerax.core.commands import Or, EmptyArg, FloatArg, Color8Arg, Float3Arg, BoolArg, ColormapArg
from chimerax.core.commands import ColormapRangeArg
from chimerax.atomic import ResiduesArg

register("label",
    CmdDesc(
        required=[('residues', Or(ResiduesArg, EmptyArg))],
        keyword=[('height', FloatArg), ('no_data_color', Color8Arg), ('offset', Float3Arg),
            ('on_top', BoolArg), ('palette', ColormapArg), ('range', ColormapRangeArg)],
        synopsis='Label residues with grid data'),
    label_cmd, registry=registry)

def run(session, pg, text):
    from chimerax.core.commands import Command
    cmd = Command(session, registry=registry)
    global _pg
    _pg = pg
    try:
        cmd.run(text, log=False)
    finally:
        _pg = None
