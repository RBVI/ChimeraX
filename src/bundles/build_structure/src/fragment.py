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

from chimerax.core.toolshed import ProviderManager

class Fragment:
    def __init__(self, name, atoms, bonds):
        """
        *name* is the fragment name (e.g. "benzene")

        *atoms* is a list of tuples: (element name, xyz)

        *bonds* is a list of tuples: (indices, depict)
        where *indices* is a two-tuple into the atom list and *depict* is either None [single bond]
        or an xyz [center of ring] for a double bond.  Depiction of non-ring double bonds not supported yet.
        File a bug report if you need them.
        """
        self.name = name
        self.atoms = atoms
        self.bonds = bonds

    def depict(self, scene, scale):
        from Qt.QtGui import QFont, QFontMetrics
        font = QFont('Helvetica', scale-1)

        depicted_hydrogens = []
        text_info = {}
        for indices, double in self.bonds:
            atoms = [self.atoms[i] for i in indices]
            elements = [a[0] for a in atoms]
            if "H" in elements:
                if "C" not in elements:
                    h_place = elements.index("H")
                    depicted_hydrogens.append((indices[h_place], indices[1-h_place]))
                continue
            scene.addLine(*qt_coords(atoms, scale))

            if double:
                double_bond = []
                for a in atoms:
                    for i in range(2):
                        double_bond.append(scale * (0.8*a[1][i] + 0.2 * double[i]))
                scene.addLine(*qt_coords(double_bond))

        for a in self.atoms:
            element = a[0]
            if element in "HC" or a in text_info:
                continue
            atom_pos = qt_coords([a], scale)
            text_info[a] = draw_text(scene, element, font, atom_pos)

        h_rect = QFontMetrics(font).boundingRect("H")
        for hi, oi in depicted_hydrogens:
            h = self.atoms[hi]
            o = self.atoms[oi]
            xdiff = o[1][0] - h[1][0]
            ydiff = o[1][1] - h[1][1]
            x, y = qt_coords([o], scale)
            b_rect = text_info[o]
            if abs(xdiff) > abs(ydiff):
                if xdiff < 0:
                    # H to the right
                    x += b_rect.width()
                else:
                    x -= h_rect.width()
            else:
                if ydiff < 0:
                    # H above
                    y -= b_rect.height()
                else:
                    y += h_rect.height()
            draw_text(scene, "H", font, (x,y), background=False)

def draw_text(scene, text, font, pos, background=True):
    txt = scene.addSimpleText(text, font)
    txt.setZValue(2)
    b_rect = txt.boundingRect()
    x, y = pos[0] - b_rect.width()/2, pos[1] -  b_rect.height()/2
    txt.setPos(x, y)
    if background:
        from Qt.QtGui import QPen, QBrush
        from Qt.QtCore import Qt
        pen = QPen(Qt.NoPen)
        brush = QBrush(scene.backgroundBrush())
        backing = scene.addRect(b_rect, pen, brush)
        backing.setZValue(1)
        backing.setPos(x, y)
    return b_rect

def qt_coords(source, scale=None):
    coords = []
    if scale is None:
        for i in range(0, len(source), 2):
            coords.append(source[i])
            coords.append(0 - source[i+1])
    else:
        for a in source:
            coords.append(scale * a[1][0])
            coords.append(0 - scale * a[1][1])
    return tuple(coords)
