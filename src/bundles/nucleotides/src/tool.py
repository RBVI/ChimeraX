# vim: set expandtab shiftwidth=4 softtabstop=4:

# --- UCSF Chimera Copyright ---
# Copyright (c) 2004 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  This notice must be embedded in or
# attached to all copies, including partial copies, of the
# software or any revisions or derivations thereof.
# --- UCSF Chimera Copyright ---

from chimera.baseDialog import ModelessDialog
from chimera.tkoptions import Option, EnumOption, FloatOption, BooleanOption
import NucleicAcids as NA
import default
from chimerax.core.tools import ToolInstance
import chimera, Tk, Pmw


class NucleotidesTool(ToolInstance):

    SESSION_ENDURING = False
    SESSION_SKIP = True         # No session saving for now
    display_name = "NucleotidesTool"

    def __init__(self, session, tool_name):
        # Standard template stuff for intializing tool
        super().__init__(session, tool_name)
        from chimerax.core.ui.gui import MainToolWindow
        self.tool_window = MainToolWindow(self)
        self.tool_window.manage(placement="side")
        parent = self.tool_window.ui_area

        # Create an HTML viewer for our user interface.
        # We can include other Qt widgets if we want to.
        from PyQt5.QtWidgets import QGridLayout
        from chimerax.core.ui.widgets import HtmlView
        layout = QGridLayout()
        self.html_view = HtmlView(parent, size_hint=(575, 200))
        layout.addWidget(self.html_view, 0, 0)  # row 0, column 0
        parent.setLayout(layout)

    def delete(self):
        pass


class BackboneOption(EnumOption):
        values = ['atoms & bonds', 'ribbon']


class SideOption(EnumOption):
        values = [
            'atoms & bonds', 'fill/fill', 'fill/slab', 'tube/slab', 'ladder'
        ]


class ShapeOption(EnumOption):
        values = ['box', 'tube', 'ellipsoid']


class AnchorOption(EnumOption):
        values = [NA.SUGAR, NA.BASE]


class Float2Option(Option):
        """Specialization for (x, y) input"""

        min = None
        max = None
        cbmode = Option.RETURN

        def _addOption(self, min=None, max=None, **kw):
                if min is not None:
                        self.min = min
                if max is not None:
                        self.max = max
                entry_opts = {
                    'validatecommand': self._val_register(self._set),
                    'width': 8,
                    'validate': 'all',
                }
                entry_opts.update(kw)
                self._option = Tk.Frame(self._master)
                self.entries = []
                for i in range(2):
                        e = Tk.Entry(self._option, **entry_opts)
                        e.pack(side=Tk.TOP)
                        self.entries.append(e)
                self.bgcolor = self.entries[0].cget('bg')
                self._value = [None, None]
                return self._option

        def enable(self):
                for e in self.entries:
                        e.config(state=Tk.NORMAL)

        def disable(self):
                for e in self.entries:
                        e.config(state=Tk.DISABLED)

        def _set(self, args):
                action = args['action']
                w = args['widget']
                index = self.entries.index(w)
                entry = self.entries[index]
                try:
                        value = float(args['new'])
                except ValueError:
                        if action != -1:
                                entry.configure(bg=self.errorColor)
                        else:
                                # enter/leave, reset to valid value
                                entry.configure(bg=self.bgcolor)
                                if self._value[0] and self._value[1]:
                                        self.set(self._value)
                        return Tk.TRUE
                entry.configure(bg=self.bgcolor)
                if action == -1:
                        if self.min is not None and value < self.min:
                                if self._value[index] != self.min:
                                        self._update_value(value, index)
                                        Option._set(self)
                        elif self.max is not None and value > self.max:
                                if self._value[index] != self.max:
                                        self._update_value(value, index)
                                        Option._set(self)
                        else:
                                if self._value != self.max:
                                        self._value[index] = value
                                        Option._set(self)
                        return Tk.TRUE
                if (self.min is not None and value < self.min) or \
                        (self.max is not None and value > self.max):
                    entry.configure(bg=self.errorColor)
                    return Tk.TRUE
                if self.cbmode == Option.CONTINUOUS:
                    self._value[index] = value
                    Option._set(self)
                return Tk.TRUE

        def _bindReturn(self):
                # override as appropriate
                for e in self.entries:
                        e.bind('<Return>', self._return)

        def _return(self, e=None):
                widget = e.widget
                args = {
                    'action': -1,
                    'new': widget.get(),
                    'widget': widget
                }
                self._set(args)
                if self.cbmode == Option.RETURN_TAB or \
                        (self.cbmode == Option.RETURN and widget != self.entries[1]):
                    w = widget.tk_focusNext()
                    if w:
                        w.focus()

        def _update_index(self, value, index):
                self._value[index] = value
                entry = self.entries[index]
                validate = entry.cget('validate')
                if validate != Tk.NONE:
                        entry.configure(validate=Tk.NONE)
                entry.delete(0, Tk.END)
                strvalue = '%g' % value
                if '.' not in strvalue:
                        strvalue += '.0'
                entry.insert(Tk.END, strvalue)
                if validate != Tk.NONE:
                        entry.configure(validate=validate)

        def set(self, value):
                assert(len(value) == 2)
                value = [float(value[0]), float(value[1])]
                for index in range(2):
                        self._update_index(value[index], index)

        def get(self):
                return tuple(self._value)

        def setMultiple(self):
                raise RuntimeError("%s does not implement setMultiple()" % (
                    self.__class__.__name__))


class Interface(ModelessDialog):

        title = 'Nucleotides'
        help = "ContributedSoftware/nucleotides/nucleotides.html"
        buttons = ("NDB Colors",) + ModelessDialog.buttons
        provideStatus = True

        def __init__(self, *args, **kw):
                self.currentStyle = self.saveui_defaultItem()
                ModelessDialog.__init__(self, *args, **kw)
                self.__firstcanvas = True

        def fillInUI(self, parent):
                parent.columnconfigure(0, pad=2)
                parent.columnconfigure(1, pad=2)
                import itertools
                row = itertools.count()

                self.showBackbone = BackboneOption(
                    parent, row.next(), 'Show backbone as', 'ribbon', None)
                self.showSide = SideOption(
                    parent, row.next(), 'Show side (sugar/base) as', 'fill/slab',
                    self._showSideCB)
                self.showOrientation = BooleanOption(
                    parent, row.next(), 'Show base orientation', default.ORIENT, None)

                import Tix
                self.nb = Tix.NoteBook(parent)
                self.nb.grid(row=row.next(), column=0, columnspan=2, sticky=Tk.EW, padx=2)

                # ladder page
                self.nb.add("ladder", label="Ladder Options")
                f = self.nb.page("ladder")
                if Tk.TkVersion >= 8.5:
                        parent.tk.call('grid', 'anchor', f._w, Tk.N)

                prow = itertools.count()
                self.skipNonBase = BooleanOption(
                    f, prow.next(), 'Ignore non-base H-bonds', default.IGNORE, None)
                self.showStubs = BooleanOption(
                    f, prow.next(), 'Show stubs', default.STUBS, None)
                self.rungRadius = FloatOption(
                    f, prow.next(), 'Rung radius', default.RADIUS, None)
                self.rungRadius.min = 0.0
                self.useExisting = BooleanOption(
                    f, prow.next(),
                    'Using existing H-bonds', default.USE_EXISTING,
                    self._useExistingCB)
                from FindHBond.gui import RelaxParams
                self.relaxParams = RelaxParams(f, None, colorOptions=False)
                # self.relaxParams.relaxConstraints = False
                self.relaxParams.grid(row=prow.next(), columnspan=2, sticky='nsew', padx=2)

                # slab page
                self.nb.add("slab", label="Slab Options")
                f = self.nb.page("slab")
                if Tk.TkVersion >= 8.5:
                        parent.tk.call('grid', 'anchor', f._w, Tk.N)

                prow = itertools.count()
                self.thickness = FloatOption(
                    f, prow.next(), 'Thickness', default.THICKNESS, None)
                self.thickness.min = 0.01
                self.shape = ShapeOption(
                    f, prow.next(), 'Slab object', default.SHAPE, None)
                self.hideBases = BooleanOption(
                    f, prow.next(), 'Hide base atoms', default.HIDE, None)
                self.showGlycosidic = BooleanOption(
                    f, prow.next(),
                    'Separate glycosidic bond', default.GLYCOSIDIC,
                    None)

                self.nb.add("style", label="Slab Style", raisecmd=self.map)
                # style page
                f = self.nb.page("style")
                if Tk.TkVersion >= 8.5:
                        f.tk.call('grid', 'anchor', f._w, Tk.N)

                info = NA.findStyle(self.currentStyle)
                from chimera.preferences import saveui
                f2 = Tk.Frame(f)
                self.saveui = saveui.SaveUI(f2, self)
                f2.grid(row=prow.next(), column=0, columnspan=2, sticky=Tk.EW, padx=2, pady=2)

                self.anchor = AnchorOption(f, prow.next(), 'Anchor',
                                           info[NA.ANCHOR], self._drawStyle)

                f2 = Pmw.Group(f, tag_text=NA.PURINE.title())
                f2.grid(row=prow.next(), column=0, columnspan=2, sticky=Tk.EW, padx=2)
                f2 = f2.interior()
                f2.columnconfigure(0, weight=1, pad=2, uniform='a')
                f2.columnconfigure(1, weight=1, uniform='a')
                f2.columnconfigure(2, weight=1)

                self.purine_canvas = Tk.Canvas(f2, width=1, height=1)
                r = prow.next()
                self.purine_canvas.grid(row=r, column=0, rowspan=3,
                                        sticky=Tk.NSEW, padx=2, pady=2)
                corners = info[NA.PURINE]
                self.puLL = Float2Option(f2, prow.next(), 'Lower left',
                                         corners[0], self._drawStyle, startCol=1)
                self.puUR = Float2Option(f2, prow.next(), 'Upper right',
                                         corners[1], self._drawStyle, startCol=1)

                f3 = Pmw.Group(f, tag_text="%s, %s" % (
                    NA.PYRIMIDINE.title(), NA.PSEUDO_PYRIMIDINE.title()))
                r = prow.next()
                f3.grid(row=r, column=0, columnspan=2, sticky=Tk.EW, padx=2)
                f3 = f3.interior()
                f3.columnconfigure(0, weight=1, pad=2, uniform='a')
                f3.columnconfigure(1, weight=1, uniform='a')
                f3.columnconfigure(2, weight=1)

                self.pyrimidine_canvas = Tk.Canvas(f3, width=1, height=1)
                r = prow.next()
                self.pyrimidine_canvas.grid(row=r, column=0, rowspan=3,
                                            sticky=Tk.NSEW, padx=2, pady=2)

                corners = info[NA.PYRIMIDINE]
                self.pyLL = Float2Option(f3, prow.next(), 'Lower left',
                                         corners[0], self._drawStyle, startCol=1)
                self.pyUR = Float2Option(f3, prow.next(), 'Upper right',
                                         corners[1], self._drawStyle, startCol=1)

                self.restrict = Tk.IntVar(parent)
                self.restrict.set(1)
                cb = Tk.Checkbutton(
                    parent, variable=self.restrict,
                    text="Restrict OK/Apply to current selection, if any")
                cb.grid(row=row.next(), columnspan=2)

                parent.pack(ipady=2)

                self._showSideCB()
                chimera.triggers.addHandler(NA.TRIGGER_SLAB_STYLES,
                                            self._updateStyles, None)

        def map(self, event=None):
                # need to update_idletasks so canvases in grid will have a size
                page = self.nb.raised()
                if page == 'style':
                        if self.__firstcanvas:
                                self.uiMaster().update_idletasks()
                                self._drawStyle()
                                self.__firstcanvas = False

        def _updateStyles(self, trigger, closure, arg):
                # pick up programmic changes in list of slab styles
                self.saveui.updateComboList()

        def saveui_label(self):
                return "Slab Style"

        def saveui_presetItems(self):
                return NA.SystemStyles.keys()

        def saveui_userItems(self):
                return [k for k in NA.userStyles.keys() if k is not None]

        def saveui_defaultItem(self):
                return 'long'

        def saveui_select(self, name):
                self.currentStyle = name
                self._setSlabStyle(name)

        def saveui_save(self, name):
                info = self._getInfo()
                NA.addStyle(name, info)
                self.status("Slab style \"%s\" saved" % name)
                return True     # successful

        def saveui_delete(self, name):
                NA.removeStyle(name)
                self.status("Slab style \"%s\" deleted" % name)
                self.currentStyle = None
                info = self._getInfo()
                NA.addStyle(None, info)
                return True     # successful

        def _showSideCB(self, *args):
                side = self.showSide.get()
                hasSlab = 'slab' in side
                if side.startswith('fill') or hasSlab:
                        self.showOrientation.enable()
                else:
                        self.showOrientation.disable()
                if hasSlab:
                        self.nb.raise_page('slab')
                elif side == 'ladder':
                        self.nb.raise_page('ladder')
                if side == 'tube/slab':
                        # tube connects to C1' if no slab
                        # if slab, it goes to the middle of slab
                        self.showGlycosidic.enable()
                        return
                self.showGlycosidic.disable()

        def _useExistingCB(self, *args):
                useExisting = self.useExisting.get()
                if useExisting:
                        self.relaxParams.disable()
                else:
                        self.relaxParams.enable()

        def Apply(self):
                from chimera import selection
                if not self.restrict.get() or selection.currentEmpty():
                        molecules = chimera.openModels.list(
                            modelTypes=[chimera.Molecule])
                        residues = []
                        for mol in molecules:
                                residues.extend(mol.residues)
                else:
                        residues = selection.currentResidues()
                residues = [r for r in residues if r.ribbonResidueClass.isNucleic()]
                molecules = set(r.molecule for r in residues)

                backbone = self.showBackbone.get()
                display = backbone != 'atoms & bonds'
                for r in residues:
                        r.ribbonDisplay = display

                side = self.showSide.get()
                if side == 'ladder':
                        distSlop = 0.0
                        angleSlop = 0.0
                        relax = self.relaxParams.relaxConstraints
                        if relax:
                                distSlop = self.relaxParams.relaxDist
                                angleSlop = self.relaxParams.relaxAngle
                        NA.set_ladder(molecules, residues,
                                      rungRadius=self.rungRadius.get(),
                                      showStubs=self.showStubs.get(),
                                      skipNonBaseHBonds=self.skipNonBase.get(),
                                      useExisting=self.useExisting.get(),
                                      distSlop=distSlop, angleSlop=angleSlop)
                        return
                if side.endswith('slab'):
                        if self.currentStyle is None:
                                info = self._getInfo()
                                NA.addStyle(None, info)
                        showGly = self.anchor.get() != NA.SUGAR
                        if showGly and side.startswith('tube'):
                                showGly = self.showGlycosidic.get()
                        NA.set_slab(side, molecules, residues,
                                    style=self.currentStyle,
                                    thickness=self.thickness.get(),
                                    orient=self.showOrientation.get(),
                                    shape=self.shape.get(), showGly=showGly,
                                    hide=self.hideBases.get())
                if side.startswith('fill'):
                        for r in residues:
                                r.fillDisplay = True
                else:
                        for r in residues:
                                r.fillDisplay = False
                if side.endswith('fill'):
                        if self.showOrientation.get():
                                NA.set_orient(molecules, residues)
                        else:
                                NA.set_normal(molecules, residues)
                elif side.startswith('atoms'):
                        NA.set_normal(molecules, residues)

        def _showBase(self, type, info, canvas):
                # assume height is greater than width
                # keep in mind, canvases are "left-handed", so y is inverted
                # get unique bases of given type
                unique_bases = {}
                for b in NA.standard_bases.values():
                        if b['type'] == type:
                                unique_bases[id(b)] = b
                # compute drawing parameters
                win_width = canvas.winfo_width()
                if win_width == 1:
                        # no size assigned yet
                        return
                win_height = canvas.winfo_height()

                # TODO: figure out how much room we really need for text
                if win_width < win_height:
                        win_scale = .8 * win_width
                else:
                        win_scale = .8 * win_height
                x_offset = .1 * win_width + 2   # 2==borderwidth
                if type == NA.PURINE:
                        min = NA.purine_min
                        max = NA.purine_max
                        other = NA.pyrimidine_max[0] - NA.pyrimidine_min[0]
                elif type == NA.PYRIMIDINE:
                        min = NA.pyrimidine_min
                        max = NA.pyrimidine_max
                        other = NA.purine_max[0] - NA.purine_min[0]
                width = max[0] - min[0]
                if other > width:
                        width = other
                scale = win_scale / width
                # center vertically
                height = (max[1] - min[1]) * scale
                win_height -= (win_height - height) / 2

                # clear canvas
                canvas.addtag_all('all')
                canvas.delete('all')

                def cvt_coords(c):
                        for i in range(0, len(c), 2):
                                c[i] = (c[i] - min[0]) * scale + x_offset
                                c[i + 1] = win_height \
                                    - (c[i + 1] - min[1]) * scale

                def draw_line(b, names):
                        coords = []
                        for n in names:
                                c = b[n]
                                coords += [c[0], c[1]]
                        if len(names) > 2:
                                # close line
                                coords += coords[0: 2]
                        cvt_coords(coords)
                        kw = {'width': 2}
                        canvas.create_line(*coords, **kw)
                c1p_coords = None
                for b in unique_bases.values():
                        rn = b['ring atom names']
                        draw_line(b, rn)
                        for o in b['other bonds']:
                                draw_line(b, o)
                        if c1p_coords is None:
                                c1p_coords = list(b["C1'"][0: 2])
                        else:
                                coords = b["C1'"][0: 2]
                                if coords[0] < c1p_coords[0]:
                                        c1p_coords[0] = coords[0]
                                if coords[1] > c1p_coords[1]:
                                        c1p_coords[1] = coords[1]
                corners = info[type]
                anchor = NA.anchor(info[NA.ANCHOR], type)
                offset = b[anchor]
                coords = [
                    offset[0] + corners[0][0], offset[1] + corners[0][1],
                    offset[0] + corners[1][0], offset[1] + corners[1][1]
                ]
                cvt_coords(coords)
                kw = {'fill': 'gray25', 'stipple': 'gray25'}
                canvas.create_rectangle(*coords, **kw)
                coords = [min[0], max[1]]
                cvt_coords(c1p_coords)
                kw = {'text': " C1'", 'anchor': 's'}
                canvas.create_text(*c1p_coords, **kw)

        def _getInfo(self):
                info = {
                    NA.ANCHOR: self.anchor.get(),
                    NA.PURINE: (self.puLL.get(), self.puUR.get()),
                    NA.PYRIMIDINE: (self.pyLL.get(), self.pyUR.get()),
                    NA.PSEUDO_PYRIMIDINE: (self.pyLL.get(), self.pyUR.get())
                }
                return info

        def _drawStyle(self, *args):
                if args:
                        self.saveui.setItemChanged(True)
                        self.currentStyle = None

                # fill in parameters
                info = self._getInfo()

                # show bases
                self._showBase(NA.PURINE, info, self.purine_canvas)
                self._showBase(NA.PYRIMIDINE, info, self.pyrimidine_canvas)

        def _setSlabStyle(self, name):
                # make options reflect current style
                info = NA.findStyle(name)
                if not info:
                        return
                self.currentStyle = name
                self.anchor.set(info[NA.ANCHOR])
                corners = info[NA.PURINE]
                self.puLL.set(corners[0])
                self.puUR.set(corners[1])
                corners = info[NA.PYRIMIDINE]
                self.pyLL.set(corners[0])
                self.pyUR.set(corners[1])
                self._drawStyle()

        def NDBColors(self):
                from chimera import selection
                if selection.currentEmpty():
                        import Midas
                        residues = Midas._selectedResidues('#')
                else:
                        residues = selection.currentResidues()
                NA.NDBColors(residues)


singleton = None


def gui():
        global singleton
        if not singleton:
                singleton = Interface()
        singleton.enter()
