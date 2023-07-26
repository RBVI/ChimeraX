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

# ------------------------------------------------------------------------------
#
from chimerax.core.tools import ToolInstance
class FitList(ToolInstance):

    buttons = ('Place Copy', 'Save PDB', 'Options', 'Delete', 'Clear List', 'Close')

    def __init__(self, session, tool_name):

        self.list_fits = []
        self.show_clash = False
        self.smooth_motion = True
        self.smooth_steps = 10

        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        self.tool_window = tw
        parent = tw.ui_area

        # Place list above row of buttons
        w = parent
        from Qt.QtWidgets import QVBoxLayout, QListWidget, QAbstractItemView
        vb = QVBoxLayout()
        vb.setContentsMargins(0,0,0,0)          # No border padding
        vb.setSpacing(0)                # Spacing between list and button row

        class ListBox(QListWidget):
            def sizeHint(self):
                from Qt import QtCore
                return QtCore.QSize(500,50)
            def keyPressEvent(self, event):
                # Handle copy shortcut to copy list lines to clipboard.
                from Qt.QtGui import QKeySequence, QGuiApplication
                if event.matches(QKeySequence.StandardKey.Copy):
                    lines = [item.text() for item in self.selectedItems()]
                    QGuiApplication.clipboard().setText('\n'.join(lines))
                else:
                    QListWidget.keyPressEvent(self, event)  # Handle non-copy keys like arrows

        self.list_box = lb = ListBox(w)
        lb.setSelectionMode(QAbstractItemView.ExtendedSelection)
        lb.itemSelectionChanged.connect(self.fit_selection_cb)
        vb.addWidget(lb)

        # Button row
        from Qt.QtWidgets import QWidget, QHBoxLayout, QPushButton
        buttons = QWidget(w)
        hb = QHBoxLayout()
        hb.setContentsMargins(0,0,0,0)          # No border padding
        hb.addStretch(1)                        # Stretchable space at left
        hb.setSpacing(5)                # Spacing between buttons
        for bname,cb in (('Place Copy', self.place_copies_cb),
                         ('Save PDB', self.save_fits_cb),
                         ('Delete', self.delete_fit_cb),
                         ('Clear List', lambda self=self: self.delete_fit_cb(all=True))):
            b = QPushButton(bname, buttons)
            b.pressed.connect(cb)
            hb.addWidget(b)
        buttons.setLayout(hb)
        vb.addWidget(buttons)
# Appears that on mac there is a problem that prevents reducing QPushButton border to 0.
#        b = QPushButton('Test', w)
#        b.setContentsMargins(0,0,0,0)          # No effect
#        vb.addWidget(b)

        w.setLayout(vb)

        from Qt.QtGui import QFont
        lb.setFont(QFont("Courier"))  # Fixed with font so columns line up

        self.refill_list()      # Set heading

        tw.manage(placement="side")

    # ---------------------------------------------------------------------------
    #
    @classmethod
    def get_singleton(self, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, FitList, 'Fit List', create=create)

    def show(self):
        from Qt import QtCore
        dw = self.dock_widget
        self.session.ui.main_window.addDockWidget(QtCore.Qt.TopDockWidgetArea, dw)
        dw.setVisible(True)

    def hide(self):
        self.session.ui.main_window.removeDockWidget(self.dock_widget)

    def fillInUI(self, parent):

        parent.columnconfigure(0, weight = 1)

        row = 1
    
        fl = Hybrid.Scrollable_List(parent, 'Fits', 10, self.fit_selection_cb)
        fl.heading.configure(font = 'TkFixedFont')
        fl.listbox.configure(font = 'TkFixedFont', width = 70)
        self.listbox_heading = fl.heading
        self.fit_listbox = fl.listbox
        self.list_fits = []
        fl.frame.grid(row = row, column = 0, sticky = 'news')
        parent.rowconfigure(row, weight = 1)
        row += 1
        self.fit_listbox.bind('<KeyPress-Delete>', self.delete_fit_cb)

        op = Hybrid.Popup_Panel(parent, resize_dialog = False)
        opf = op.frame
        opf.grid(row = row, column = 0, sticky = 'news')
        opf.grid_remove()
        opf.columnconfigure(0, weight=1)
        self.options_panel = op.panel_shown_variable
        row += 1
        orow = 0

        cb = op.make_close_button(opf)
        cb.grid(row = orow, column = 1, sticky = 'e')

        sm = Hybrid.Checkbutton_Entries(opf, True,
                                        'Smooth motion between fits ',
                                        (3, '10'), ' steps')
        sm.frame.grid(row = orow, column = 0, sticky = 'nw')
        orow += 1
        self.smooth_motion, self.smooth_steps = sm.variables

        cl = Hybrid.Checkbutton(opf, 'Show clash volume fraction between symmetric copies', False)
        cl.button.grid(row = orow, column = 0, sticky = 'w')
        orow += 1
        self.show_clash = cl.variable
        cl.callback(self.show_clash_cb)

        self.refill_list()      # Set heading

        from SimpleSession import SAVE_SESSION
        from chimera import triggers, CLOSE_SESSION
        triggers.addHandler(SAVE_SESSION, self.save_session_cb, None)
        triggers.addHandler(CLOSE_SESSION, self.close_session_cb, None)

        add_fit_list_menu_entry()

    def Options(self):
        self.options_panel.set(not self.options_panel.get())

    def add_fits(self, fits):

        for f in fits:
            self.add_fit(f)

    def add_fit(self, fit):

        line = self.list_line(fit)
        self.list_box.addItem(line)
        self.list_fits.append(fit)

    def heading(self):

        clash = (' %8s' % 'Clash') if self.show_clash else ''
        h = '%8s %8s %8s%s %15s %15s %5s' % ('Corr  ', 'Ave ', 'Inside',  clash,
                                             'Molecule', 'Map     ', 'Hits')
        return h

    def list_line(self, fit):

        # TODO: fit list should probably report optimization metric.
        # TODO: When fit using correlation about mean, should report correlation
        #   about mean.  Maybe report both about mean and not about mean.
        # TODO: Want to be able to update clash value when contour level
        #   changed.  Currently it is cached in Fit object.
        mname = fit.models[0].name if fit.models else 'deleted'
        from chimerax.atomic import Structure
        if len([m for m in fit.models if isinstance(m,Structure)]) > 1:
            mname += '...'

        c = fit.correlation()
        cs = ('%8s' % '') if c is None else '%8.4f' % c
        clash = ''
        if self.show_clash:
            c = fit.clash()
            clash = (' %8.3f' % c) if not c is None else (' %8s' % '')
        amv = fit.average_map_value()
        amvs = ('%8s' % '') if amv is None else '%8.3f' % amv
        pic = fit.points_inside_contour()
        pics = ('%8s' % '') if pic is None else '%8.3f' % pic
        fv = fit.volume
        vname = 'deleted' if fv is None or fv.was_deleted else fv.name
        line = '%8s %s %s%s %15s %15s %5d' % (cs, amvs, pics, clash, mname,
                                              vname, fit.hits())
        return line

    def refill_list(self):

        lb = self.list_box
        lb.clear()
        h = self.heading()
        lb.addItem(h)
        for fit in self.list_fits:
            line = self.list_line(fit)
            lb.addItem(line)

    def fit_selection_cb(self, event = None):

        lfits = self.selected_listbox_fits()
        if len(lfits) == 0:
            return

        frames = 0
        if self.smooth_motion:
            frames = self.smooth_steps

        lfits[0].place_models(self.session, frames)

    def selected_listbox_fits(self):

        lb = self.list_box
        return [f for r,f in enumerate(self.list_fits)
                if lb.item(r+1).isSelected()]

    def select_fit(self, fit):

        row = self.list_fits.index(fit)+1
        lb = self.list_box
        lb.item(row).setSelected(True)
        self.fit_selection_cb()
        
    def save_fits_cb(self):

        lfits = self.selected_listbox_fits()
        from .search import save_fits
        save_fits(self.session, lfits)

    def delete_fit_cb(self, all = False):

        if all:
            dfits = self.list_fits
        else:
            dfits = self.selected_listbox_fits()
            if len(dfits) == 0:
                self.session.logger._status('No fits chosen from list.')
                return
        dset = set(dfits)
        fits = self.list_fits
        indices = [i for i,f in enumerate(fits) if f in dset]
        indices.reverse()
        lb = self.list_box
        for i in indices:
            lb.takeItem(i+1)
            del fits[i]

        self.session.logger.status('Deleted %d fits' % len(indices))

    def place_copies_cb(self):

        lfits = [f for f in self.selected_listbox_fits() if f.fit_molecules()]
        if len(lfits) == 0:
            self.session.logger.status('No fits of molecules chosen from list.')
            return
        clist = []
        for fit in lfits:
            clist.extend(fit.place_copies())
        self.session.logger.status('Placed %d molecule copies' % len(clist))

    def show_clash_cb(self):

        self.refill_list()

    def save_session_cb(self, trigger, x, file):

        if self.list_fits:
            import session
            session.save_fit_list_state(self, file)

    def close_session_cb(self, trigger, x, y):

        self.delete_fit_cb(all = True)

        
# -----------------------------------------------------------------------------
#
def fit_list_dialog(session, create = False):

    return FitList.get_singleton(session, create=create)
  
# -----------------------------------------------------------------------------
#
def show_fit_list_dialog(session):

    return fit_list_dialog(session, create = True)
