from chimera.replyobj import status, warning

# ------------------------------------------------------------------------------
#
from chimera.baseDialog import ModelessDialog
class Fit_List(ModelessDialog):

    title = 'Fit List'
    name = 'fit list'
    buttons = ('Place Copy', 'Save PDB', 'Options', 'Delete', 'Clear List', 'Close')
    help = 'UsersGuide/midas/fitmap.html#fitlist'

    def fillInUI(self, parent):

        import Tkinter
        from CGLtk import Hybrid

        tw = parent.winfo_toplevel()
        self.toplevel_widget = tw
        tw.withdraw()

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

    def map(self):
        # Can get scrollbar map/unmap infinite loop if dialog resize allowed.
        tw = self.toplevel_widget
        tw.geometry(tw.geometry())

    def PlaceCopy(self):
        self.place_copies_cb()

    def SavePDB(self):
        self.save_fits_cb()

    def Options(self):
        self.options_panel.set(not self.options_panel.get())

    def Delete(self):
        self.delete_fit_cb()

    def ClearList(self):
        self.delete_fit_cb(all = True)

    def add_fits(self, fits):

        for f in fits:
            self.add_fit(f)

    def add_fit(self, fit):

        line = self.list_line(fit)
        self.fit_listbox.insert('end', line)
        self.list_fits.append(fit)

    def heading(self):

        clash = (' %8s' % 'Clash') if self.show_clash.get() else ''
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
        from chimera import Molecule
        if len([m for m in fit.models if isinstance(m,Molecule)]) > 1:
            mname += '...'

        c = fit.correlation()
        cs = ('%8s' % '') if c is None else '%8.4f' % c
        clash = ''
        if self.show_clash.get():
            c = fit.clash()
            clash = (' %8.3f' % c) if not c is None else (' %8s' % '')
        amv = fit.average_map_value()
        amvs = ('%8s' % '') if amv is None else '%8.3f' % amv
        pic = fit.points_inside_contour()
        pics = ('%8s' % '') if pic is None else '%8.3f' % pic
        fv = fit.volume
        vname = 'deleted' if fv is None or fv.__destroyed__ else fv.name
        line = '%8s %s %s%s %15s %15s %5d' % (cs, amvs, pics, clash, mname,
                                              vname, fit.hits())
        return line

    def refill_list(self):

        lbox = self.fit_listbox
        lbox.delete('0', 'end')
        self.listbox_heading['text'] = self.heading()
        for fit in self.list_fits:
            lbox.insert('end', self.list_line(fit))
            
    def fit_selection_cb(self, event = None):

        lfits = self.selected_listbox_fits()
        if len(lfits) == 0:
            return

        frames = 0
        if self.smooth_motion.get():
            from CGLtk import Hybrid
            frames = Hybrid.integer_variable_value(self.smooth_steps, 0, 0)

        lfits[0].place_models(frames)

    def selected_listbox_fits(self):

        return [self.list_fits[int(i)] for i in self.fit_listbox.curselection()]

    def select_fit(self, fit):

        self.fit_listbox.selection_set(self.list_fits.index(fit))
        self.fit_selection_cb()
        
    def save_fits_cb(self):

        lfits = self.selected_listbox_fits()
        mlist = sum([f.fit_molecules() for f in lfits], [])
        if len(mlist) == 0:
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
                        fit.place_models()
                        Midas.write(fit.fit_molecules(), relModel = fit.volume,
                                    filename = p)
                      
        from OpenSave import SaveModeless
        SaveModeless(title = 'Save Fit Molecules',
                     filters = [('PDB', '*.pdb', '.pdb')],
                     initialdir = idir, initialfile = ifile, command = save)

    def delete_fit_cb(self, all = False):

        if all:
            indices = range(len(self.list_fits))
        else:
            indices = [int(i) for i in self.fit_listbox.curselection()]
            if len(indices) == 0:
                warning('No fits chosen from list.')
                return
            indices.sort()
        indices.reverse()
        for i in indices:
            self.fit_listbox.delete(i)
            del self.list_fits[i]

        status('Deleted %d fits' % len(indices))

    def place_copies_cb(self):

        lfits = [f for f in self.selected_listbox_fits() if f.fit_molecules()]
        if len(lfits) == 0:
            warning('No fits of molecules chosen from list.')
            return
        clist = []
        for fit in lfits:
            clist.extend(fit.place_copies())
        status('Placed %d molecule copies' % len(clist))

    def show_clash_cb(self):

        self.refill_list()

    def save_session_cb(self, trigger, x, file):

        if self.list_fits:
            import session
            session.save_fit_list_state(self, file)

    def close_session_cb(self, trigger, x, y):

        self.ClearList()

# -----------------------------------------------------------------------------
#
added_fit_list_menu_entry = False
def add_fit_list_menu_entry():

    global added_fit_list_menu_entry
    if added_fit_list_menu_entry:
        return
    added_fit_list_menu_entry = True

    from chimera.extension import EMO, manager
    class Fit_List_EMO(EMO):
        def name(self):
            return 'Fit List'
        def description(self):
            return 'List of fits in maps created by fitmap search command'
        def categories(self):
            return ['Volume Data']
        def icon(self):
            return None
        def activate(self):
            show_fit_list_dialog()
            return None
    manager.registerExtension(Fit_List_EMO(__file__))

    # Update menus
    manager.remakeCategoryMenu(manager.findCategory('Volume Data'))
    import VolumeMenu
    VolumeMenu.remake_toplevel_volume_menu()
    from VolumeViewer.volumedialog import volume_dialog
    d = volume_dialog()
    if d:
        d.update_tools_menu()
        
# -----------------------------------------------------------------------------
#
def fit_list_dialog(create = False):

    from chimera import dialogs
    return dialogs.find(Fit_List.name, create=create)
  
# -----------------------------------------------------------------------------
#
def show_fit_list_dialog():

    from chimera import dialogs
    d = dialogs.display(Fit_List.name)
    d.fit_listbox.focus_set()
    return d

# -----------------------------------------------------------------------------
#
from chimera import dialogs
dialogs.register(Fit_List.name, Fit_List, replace = True)
