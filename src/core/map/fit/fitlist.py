# ------------------------------------------------------------------------------
#
class Fit_List:

    buttons = ('Place Copy', 'Save PDB', 'Options', 'Delete', 'Clear List', 'Close')

    def __init__(self, session):

        self.session = session
        self.list_fits = []
        self.show_clash = False
        self.smooth_motion = True
        self.smooth_steps = 10

        from ...ui.qt import QtWidgets
        self.dock_widget = dw = QtWidgets.QDockWidget('Fit List', session.main_window)

        # Place list above row of buttons
        w = QtWidgets.QWidget(dw)
        vb = QtWidgets.QVBoxLayout()
        vb.setContentsMargins(0,0,0,0)          # No border padding
        vb.setSpacing(0)                # Spacing between list and button row
        class ListBox(QtWidgets.QListWidget):
            def sizeHint(self):
                from ...ui.qt import QtCore
                return QtCore.QSize(500,50)
        self.list_box = lb = ListBox(w)
#        self.list_box = lb = QtWidgets.QListWidget(w)
        lb.itemSelectionChanged.connect(self.fit_selection_cb)
        vb.addWidget(lb)

        # Button row
        buttons = QtWidgets.QWidget(w)
        hb = QtWidgets.QHBoxLayout()
        hb.setContentsMargins(0,0,0,0)          # No border padding
        hb.addStretch(1)                        # Stretchable space at left
        hb.setSpacing(5)                # Spacing between buttons
        for bname,cb in (('Place Copy', self.place_copies_cb),
                         ('Save PDB', self.save_fits_cb),
                         ('Delete', self.delete_fit_cb),
                         ('Clear List', lambda self=self: self.delete_fit_cb(all=True))):
            b = QtWidgets.QPushButton(bname, buttons)
            b.pressed.connect(cb)
            hb.addWidget(b)
        buttons.setLayout(hb)
        vb.addWidget(buttons)
# Appears that on mac there is a problem that prevents reducing QPushButton border to 0.
#        b = QtWidgets.QPushButton('Test', w)
#        b.setContentsMargins(0,0,0,0)          # No effect
#        vb.addWidget(b)

        w.setLayout(vb)
        dw.setWidget(w)

        from ...ui.qt import QtGui
        lb.setFont(QtGui.QFont("Courier"))  # Fixed with font so columns line up

        self.refill_list()      # Set heading

    def show(self):
        from ...ui.qt import QtCore
        dw = self.dock_widget
        self.session.main_window.addDockWidget(QtCore.Qt.TopDockWidgetArea, dw)
        dw.setVisible(True)

    def hide(self):
        self.session.main_window.removeDockWidget(self.dock_widget)

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
        from ...atomic import AtomicStructure
        if len([m for m in fit.models if isinstance(m,AtomicStructure)]) > 1:
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
        mlist = sum([f.fit_molecules() for f in lfits], [])
        if len(mlist) == 0:
            from chimera.replyobj import warning
            warning('No fits of molecules chosen from list.')
            return

        idir = ifile = None
        vlist = [f.volume for f in lfits]
        pmlist = [m for m in mlist + vlist if hasattr(m, 'openedAs')]
        if pmlist:
            for m in pmlist:
                import os.path
                dpath, fname = os.path.split(m.openedAs[0])
                base, suf = os.path.splitext(fname)
                if ifile is None:
                    suffix = '_fit%d.pdb' if len(lfits) > 1 else '_fit.pdb'
                    ifile = base + suffix
                if dpath and idir is None:
                    idir = dpath
            
        def save(okay, dialog, lfits = lfits):
            if okay:
                paths = dialog.getPaths()
                if paths:
                    path = paths[0]
                    import Midas
                    if len(lfits) > 1 and path.find('%d') == -1:
                        base, suf = os.path.splitext(path)
                        path = base + '_fit%d' + suf
                    for i, fit in enumerate(lfits):
                        p = path if len(lfits) == 1 else path % (i+1)
                        fit.place_models(self.session)
                        Midas.write(fit.fit_molecules(), relModel = fit.volume,
                                    filename = p)
                      
        from OpenSave import SaveModeless
        SaveModeless(title = 'Save Fit Molecules',
                     filters = [('PDB', '*.pdb', '.pdb')],
                     initialdir = idir, initialfile = ifile, command = save)

    def delete_fit_cb(self, all = False):

        if all:
            dfits = self.list_fits
        else:
            dfits = self.selected_listbox_fits()
            if len(dfits) == 0:
                self.session.show_status('No fits chosen from list.')
                return
        dset = set(dfits)
        fits = self.list_fits
        indices = [i for i,f in enumerate(fits) if f in dset]
        indices.reverse()
        lb = self.list_box
        for i in indices:
            lb.takeItem(i+1)
            del fits[i]

        self.session.show_status('Deleted %d fits' % len(indices))

    def place_copies_cb(self):

        lfits = [f for f in self.selected_listbox_fits() if f.fit_molecules()]
        if len(lfits) == 0:
            self.session.show_status('No fits of molecules chosen from list.')
            return
        clist = []
        for fit in lfits:
            clist.extend(fit.place_copies())
        self.session.show_status('Placed %d molecule copies' % len(clist))

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
    fit_list = session.fit_list
    if fit_list is None and create:
        session.fit_list = fit_list = Fit_List(session)
    return fit_list
  
# -----------------------------------------------------------------------------
#
def show_fit_list_dialog(session):

    fl = fit_list_dialog(session, create = True)
    fl.show()
    return fl
