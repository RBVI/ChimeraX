# --- UCSF Chimera Copyright ---
# Copyright (c) 2000 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  This notice must be embedded in or
# attached to all copies, including partial copies, of the
# software or any revisions or derivations thereof.
# --- UCSF Chimera Copyright ---
#
# $Id: clustalX.py 26655 2009-01-07 22:02:30Z gregc $

_clustal_red     = (0.9, 0.2, 0.1)
_clustal_blue    = (0.1, 0.5, 0.9)
_clustal_green   = (0.1, 0.8, 0.1)
_clustal_cyan    = (0.1, 0.7, 0.7)
_clustal_pink    = (0.9, 0.5, 0.5)
_clustal_magenta = (0.8, 0.3, 0.8)
#_clustal_yellow  = (0.8, 0.8, 0.0)
# above is very hard to see on the gray background, so...
_clustal_yellow  = (0.69, 0.69, 0.0)
_clustal_orange  = (0.9, 0.6, 0.3)
_clustal_categories = [("wlvimafcyhp", 0.6, '%'),
        ("wlvimafcyhp", 0.8, '#'), ("ed", 0.5, '-'),
        ("kr", 0.6, "+"), ("g", 0.5, 'g'), ("n", 0.5, 'n'),
        ("qe", 0.5, 'q'), ("p", 0.5, 'p'), ("ts", 0.5, 't')]
for c in "acdefghiklmnpqrstvwy":
    _clustal_categories.append((c, 0.85, c.upper()))
_clustal_colorings = {
    'G': [(_clustal_orange, None)],
    'P': [(_clustal_yellow, None)],
    'T': [(_clustal_green, "tST%#")],
    'S': [(_clustal_green, "tST#")],
    'N': [(_clustal_green, "nND")],
    'Q': [(_clustal_green, "qQE+KR")],
    'W': [(_clustal_blue, "%#ACFHILMVWYPp")],
    'L': [(_clustal_blue, "%#ACFHILMVWYPp")],
    'V': [(_clustal_blue, "%#ACFHILMVWYPp")],
    'I': [(_clustal_blue, "%#ACFHILMVWYPp")],
    'M': [(_clustal_blue, "%#ACFHILMVWYPp")],
    'A': [(_clustal_blue, "%#ACFHILMVWYPpTSsG")],
    'F': [(_clustal_blue, "%#ACFHILMVWYPp")],
    'C': [(_clustal_blue, "%#AFHILMVWYSPp"), (_clustal_pink, "C")],
    'H': [(_clustal_cyan, "%#ACFHILMVWYPp")],
    'Y': [(_clustal_cyan, "%#ACFHILMVWYPp")],
    'E': [(_clustal_magenta, "-DEqQ")],
    'D': [(_clustal_magenta, "-DEnN")],
    'K': [(_clustal_red, "+KRQ")],
    'R': [(_clustal_red, "+KRQ")]
}

def clustal_info(file_name=None):
    if file_name is None:
        return _clustal_categories, _clustal_colorings

    raise NotImplementedError("Color schemes from files not yet implemented")
    from prefs import RC_HYDROPHOBICITY
    if file_name == RC_HYDROPHOBICITY:
        import os.path
        file_name = os.path.join(os.path.dirname(__file__),
                            "kdHydrophob.par")
    colorInfo = {}
    for colorName in [ "RED", "BLUE", "GREEN", "CYAN",
                    "PINK", "MAGENTA", "YELLOW", "ORANGE"]:
        colorInfo[colorName] = eval("_clustal%s"
                        % colorName.capitalize())
    from OpenSave import osOpen
    from chimera import UserError
    f = osOpen(file_name)
    section = None
    colorSeen = False
    categories = []
    colorings = {}
    for line in f:
        line = line.strip()
        if line.startswith("@"):
            section = line[1:].lower()
            continue
        if not line:
            continue
        if section == "rgbindex":
            try:
                name, sr, sg, sb = line.split()
            except ValueError:
                raise UserError("Line in @rgbindex section of"
                    " %s is not color name followed by"
                    " red, green and blue values: '%s'"
                    % (file_name, line))
            try:
                r, g, b = [float(x) for x in [sr, sg, sb]]
            except ValueError:
                raise UserError("Line in @rgbindex section of"
                    " %s has non-floating-point"
                    " red, green or blue value: '%s'"
                    % (file_name, line))
            if r>1 or g>1 or b>1 or r<0 or g<0 or b<0:
                raise UserError("Line in @rgbindex section of"
                    " %s has red, green or blue value"
                    " not in the range 0-1: '%s'"
                    % (file_name, line))
            colorInfo[name] = (r, g, b)
        elif section == "consensus":
            try:
                symbol, eq, percent, composition = line.split()
            except ValueError:
                raise UserError("Line in @consensus section of"
                    " %s is not of the form 'symbol = "
                    " percentage%% res-list: '%s'"
                    % (file_name, line))
            if eq != '=':
                raise UserError("Line in @consensus section of"
                    " %s doesn't have '=' as second"
                    " component: '%s'" % (file_name, line))
            if percent[-1] != '%':
                raise UserError("Line in @consensus section of"
                    " %s doesn't have '%' as last character"
                    " of third component: '%s'"
                    % (file_name, line))
            try:
                percentage = float(percent[:-1])
            except ValueError:
                raise UserError("Line in @consensus section of"
                    " %s doesn't have a number before the"
                    " '%' of third component: '%s'"
                    % (file_name, line))
            if percentage < 0 or percentage > 100:
                raise UserError("Line in @consensus section of"
                    " %s has a percentage not in the range"
                    " 0-100: '%s'" % (file_name, line))
            composition = composition.replace(":", "")
            categories.append((composition, percentage/100.0,
                                symbol))
        elif section == "color":
            colorSeen = True
            fields = line.split()
            if len(fields) not in [3,5]:
                raise UserError("Line in @color section of"
                    " %s not of the form AA = color"
                    " [if consensus-list]: '%s'"
                    % (file_name, line))
            aa, eq, color = fields[:3]
            if len(aa) > 1 or not aa.islower():
                raise UserError("Line in @color section of"
                    " %s uses amino-acid code that is not"
                    " a single lowercase character: '%s'"
                    % (file_name, line))
            if eq != '=':
                raise UserError("Line in @color section of"
                    " %s doesn't have '=' as second"
                    " component: '%s'" % (file_name, line))
            if color not in colorInfo:
                raise UserError("Line in @color section of"
                    " %s uses an unknown color:"
                    " '%s'" % (file_name, line))
            if len(fields) == 3:
                colorings.setdefault(aa.upper(), []).append(
                        (colorInfo[color], None))
                continue
            if fields[3] != 'if':
                raise UserError("Line in @color section of"
                    " %s doesn't have 'if' as fourth"
                    " component: '%s'" % (file_name, line))
            colorings.setdefault(aa.upper(), []).append(
                (colorInfo[color], fields[-1].replace(":", "")))
    f.close()
    if not colorSeen:
        raise UserError("'%s' has missing or empty @color section" % file_name)
    return categories, colorings

'''
from OpenSave import OpenModeless
class ColorSchemeDialog(OpenModeless):
    """Dialog to open ClustalX-style coloring file"""

    title = "Load Residue-Letter Color Scheme"

    def __init__(self, mav):
        self.mav = mav
        OpenModeless.__init__(self, clientPos='s',
                        historyID="MAV residue colors")

    def fillInUI(self, parent):
        OpenModeless.fillInUI(self, parent)
        import Tkinter
        self.defaultVar = Tkinter.IntVar(parent)
        self.defaultVar.set(True)
        Tkinter.Checkbutton(self.clientArea, variable=self.defaultVar,
            text="Make this scheme the default").grid()

    def destroy(self):
        self.mav = None
        from chimera.baseDialog import ModelessDialog
        ModelessDialog.destroy(self)

    def Apply(self):
        if not self.getPaths():
            from chimera import replyobj
            replyobj.error("No coloring file specified.\n")
            self.enter()
            return
        self.mav.useColoringFile(self.getPaths()[0],
                    makeDefault=self.defaultVar.get())
'''
