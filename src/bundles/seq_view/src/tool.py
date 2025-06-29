# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.tools import ToolInstance
class SequenceViewer(ToolInstance):
    """ Viewer displays a multiple sequence alignment """

    help = "help:user/tools/sequenceviewer.html"

    MATCHED_REGION_INFO = ("matched residues", (1, .88, .8), "orange red")

    ERROR_REGION_STRING = "mismatches"
    GAP_REGION_STRING = "missing structure"

    """TODO
    buttons = ('Quit', 'Hide')
    help = "ContributedSoftware/multalignviewer/framemav.html"
    provideStatus = True
    statusWidth = 15
    statusPosition = "left"
    provideSecondaryStatus = True
    secondaryStatusPosition = "left"

    ConsAttr = "svPercentConserved"

    MATCH_REG_NAME_START = "matches"
    ERROR_REG_NAME_START = "mismatches"
    GAP_REG_NAME_START = "missing structure"

    # so Model Loops tool can invoke it...
    MODEL_LOOPS_MENU_TEXT = "Modeller (loops/refinement)..."

    def __init__(self, fileNameOrSeqs, fileType=None, autoAssociate=True,
                title=None, quitCB=None, frame=None, numberingDisplay=None,
                sessionSave=True):
    """
    def __init__(self, session, tool_name, alignment=None):
        """ if 'alignment' is None, then we are being restored from a session and
            _finalize_init will be called later.
        """

        ToolInstance.__init__(self, session, tool_name)
        if alignment is None:
            return
        self._finalize_init(alignment, from_session=False)

    def _finalize_init(self, alignment, *, from_session=True):
        """TODO
        from chimera import triggerSet
        self.triggers = triggerSet.TriggerSet()
        self.triggers.addTrigger(ADD_ASSOC)
        self.triggers.addTrigger(DEL_ASSOC)
        self.triggers.addTrigger(MOD_ASSOC)
        self.triggers.addHandler(ADD_ASSOC, self._fireModAssoc, None)
        self.triggers.addHandler(DEL_ASSOC, self._fireModAssoc, None)
        self.triggers.addTrigger(ADD_SEQS)
        self.triggers.addTrigger(PRE_DEL_SEQS)
        self.triggers.addTrigger(DEL_SEQS)
        self.triggers.addTrigger(ADDDEL_SEQS)
        self.triggers.addTrigger(SEQ_RENAMED)
        self.triggers.addHandler(ADD_SEQS, self._fireAddDelSeq, None)
        self.triggers.addHandler(DEL_SEQS, self._fireAddDelSeq, None)
        self.triggers.addHandler(ADDDEL_SEQS, self._fireModAlign, None)
        self.triggers.addTrigger(MOD_ALIGN)
        self.associations = {}
        self._resAttrs = {}
        self._edited = False

        from common import getStaticSeqs
        seqs, fileMarkups, fileAttrs = getStaticSeqs(fileNameOrSeqs, fileType=fileType)
        self.seqs = seqs
        """
        self.alignment = alignment
        from . import subcommand_name
        alignment.attach_viewer(self, subcommand_name=subcommand_name)
        from . import settings
        self.settings = settings.init(self.session)
        """
        from SeqCanvas import shouldWrap
        if numberingDisplay:
            defaultNumbering = numberingDisplay
        else:
            defaultNumbering = (True,
                not shouldWrap(len(seqs), self.prefs))
        self.numberingsStripped = False
        if getattr(seqs[0], 'numberingStart', None) is None:
            # see if sequence names imply numbering...
            startInfo = []
            for seq in seqs:
                try:
                    name, numbering = seq.name.rsplit('/',1)
                except ValueError:
                    break
                try:
                    start, end = numbering.split('-')
                except ValueError:
                    start = numbering
                try:
                    startInfo.append((name, int(start)))
                except ValueError:
                    break
            if len(startInfo) == len(seqs):
                self.numberingsStripped = True
                for i, seq in enumerate(seqs):
                    seq.name, seq.numberingStart = \
                                startInfo[i]
            else:
                for seq in seqs:
                    if hasattr(seq, 'residues'):
                        for i, r in enumerate(seq.residues):
                            if r:
                                seq.numberingStart = r.id.position - 1
                                break
                        else:
                            seq.numberingStart = 1
                    else:
                        seq.numberingStart = 1
                if not numberingDisplay:
                    defaultNumbering = (False, False)

        self._defaultNumbering = defaultNumbering
        self.fileAttrs = fileAttrs
        self.fileMarkups = fileMarkups
        if not title:
            if isinstance(fileNameOrSeqs, basestring):
                title = os.path.split(fileNameOrSeqs)[1]
            else:
                title = "Sequence Viewer"
        self.title = title
        self.autoAssociate = autoAssociate
        self.quitCB = quitCB
        self.sessionSave = sessionSave
        self._runModellerWSList = []
        self._runModellerLocalList = []
        self._realignmentWSJobs = {'self': [], 'new': []}
        self._blastAnnotationServices = {}
        ModelessDialog.__init__(self)
        """
        from chimerax.core.utils import titleize
        self.display_name = titleize(self.alignment.description) + " [ID: %s]" % self.alignment.ident
        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self, close_destroys=True, statusbar=True)
        self.tool_window._dock_widget.setMouseTracking(True)
        self.tool_window.fill_context_menu = self.fill_context_menu
        parent = self.tool_window.ui_area
        parent.setMouseTracking(True)
        """TODO
        # SeqCanvas will use these...
        leftNums, rightNums = self._defaultNumbering
        self.leftNumberingVar = Tkinter.IntVar(parent)
        self.leftNumberingVar.set(leftNums)
        self.rightNumberingVar = Tkinter.IntVar(parent)
        self.rightNumberingVar.set(rightNums)

        """
        from .seq_canvas import SeqCanvas
        self.seq_canvas = SeqCanvas(parent, self, self.alignment)
        if self.alignment.associations:
            # There are pre-existing associations, show them
            for aseq in self.alignment.seqs:
                if self.alignment.match_maps[aseq]:
                    self.seq_canvas.assoc_mod(aseq)
        self._feature_browsers = {}
        self._regions_tool = None
        from .region_browser import RegionManager
        self.region_manager = RegionManager(self.seq_canvas)
        self._seq_rename_handlers = {}
        for seq in self.alignment.seqs:
            self._seq_rename_handlers[seq] = seq.triggers.add_handler("rename",
                self.region_manager._seq_renamed_cb)
            if self.alignment.match_maps[seq]:
               self._update_errors_gaps(seq)
        if self.alignment.intrinsic and not from_session:
            self.show_ss(True)
            self.status("Helices/strands depicted in gold/green")
        if not from_session:
            if len(self.alignment.seqs) == 1:
                seq = self.alignment.seqs[0]
                if seq.features(fetch=False):
                    self.show_feature_browser(seq)
        """TODO
        if self.fileMarkups:
            from HeaderSequence import FixedHeaderSequence
            headers = []
            for name, val in self.fileMarkups.items():
                headers.append(
                    FixedHeaderSequence(name, self, val))
            self.addHeaders(headers)
        self.prefDialog = PrefDialog(self)
        top = parent.winfo_toplevel()
        cb = lambda e, rb=self.regionBrowser: rb.deleteRegion(
                                rb.curRegion())
        top.bind('<Delete>', cb)
        top.bind('<BackSpace>', cb)
        self.menuBar = Tkinter.Menu(top, type="menubar", tearoff=False)
        top.config(menu=self.menuBar)

        self.fileMenu = Tkinter.Menu(self.menuBar)
        self.menuBar.add_cascade(label="File", menu=self.fileMenu)
        self.fileMenu.add_command(label="Save As...", command=self.save)
        self.epsDialog = None
        self.fileMenu.add_command(label="Save EPS...",
                        command=self._showEpsDialog)
        self.fileMenu.add_command(label="Save Association Info...",
            state='disabled', command=self._showAssocInfoDialog)
        self.fileMenu.add_separator()
        self.fileMenu.add_command(label="Load SCF/Seqsel File...",
                command=lambda: self.loadScfFile(None))
        self.fileMenu.add_command(label="Load Color Scheme...",
                    command=self._showColorSchemeDialog)
        self.fileMenu.add_separator()
        self.fileMenu.add_command(label="Hide", command=self.Hide)
        self.fileMenu.add_command(label="Quit", command=self.Quit)
        if parent == dialogParent:
            # if we're not part of a custom interface,
            # override the window-close button to quit, not hide
            top.protocol('WM_DELETE_WINDOW', self.Quit)

        self.editMenu = Tkinter.Menu(self.menuBar)
        self.menuBar.add_cascade(label="Edit", menu=self.editMenu)
        self.editMenu.add_command(label="Copy Sequence...",
                    command=self._showCopySeqDialog)
        self.editMenu.add_command(label="Reorder Sequences...",
                    command=self._showReorderDialog)
        self.editMenu.add_command(label="Insert All-Gap Columns...",
                    command=self._showInsertGapDialog)
        self.editMenu.add_command(label="Delete Sequences/Gaps...",
                    command=self._showDelSeqsGapsDialog)
        self.editMenu.add_command(label="Add Sequence...",
                    command=self._showAddSeqDialog)
        self.editMenu.add_command(label="Realign Sequences...",
                    command=self._showRealignmentDialog)
        self.editMenu.add_command(label="Alignment Annotations...",
                    command=self._showAlignAttrDialog)
        self.editMenu.add_command(label="Edit Sequence Name...",
                    command=self._showSeqNameEditDialog)
        self.editMenu.add_command(label="Show Editing Keys...",
                    command=self._showEditKeysDialog)
        self.editMenu.add_command(label=u"Region \N{RIGHTWARDS ARROW} New Window",
                    command=self.exportActiveRegion)
        self.editMenu.add_separator()
        self.editMenu.add_command(label="Find Subsequence...",
                    command=self._showFindDialog)
        self.editMenu.add_command(label="Find Regular Expression...",
                    command=self._showRegexDialog)
        self.editMenu.add_command(label="Find PROSITE Pattern...",
                    command=self._showPrositeDialog)
        self.editMenu.add_command(label="Find Sequence Name...",
                    command=self._showFindSeqNameDialog)

        self.structureMenu = Tkinter.Menu(self.menuBar)
        self.menuBar.add_cascade(label="Structure",
                            menu=self.structureMenu)
        self.structureMenu.add_command(label="Load Structures",
                        command=self._loadStructures)
        """
        """
        self.alignDialog = self.assessDialog = self.findDialog = None
        self.prositeDialog = self.regexDialog = None
        self.associationsDialog = self.findSeqNameDialog = None
        self.saveHeaderDialog = self.alignAttrDialog = None
        self.assocInfoDialog = self.loadHeaderDialog = None
        self.identityDialog = self.colorSchemeDialog = None
        self.modellerHomologyDialog = self.fetchAnnotationsDialog = None
        self.treeDialog = self.reorderDialog = self.blastPdbDialog = None
        self.delSeqsGapsDialog = self.insertGapDialog = None
        self.addSeqDialog = self.numberingsDialog = None
        self.editKeysDialog = self.copySeqDialog = None
        self.modellerLoopsDialog = self.seqNameEditDialog = None
        self.realignDialog = None
        self.structureMenu.add_command(label="Match...",
                state='disabled', command=self._showAlignDialog)
        
        self.structureMenu.add_command(label="Assess Match...",
            state='disabled', command=self._showAssessDialog)

        if len(self.seqs) <= 1:
            state = "disabled"
        else:
            state = "normal"
        self.structureMenu.add_command(label="Modeller (homology)...",
            state=state, command=self._showModellerHomologyDialog)
        self.structureMenu.add_command(label=self.MODEL_LOOPS_MENU_TEXT,
            state="disabled", command=self._showModellerLoopsDialog)

        if chimera.openModels.list(modelTypes=[chimera.Molecule]):
            assocState = 'normal'
        else:
            assocState = 'disabled'
        self.structureMenu.add_command(label="Associations...",
            state=assocState, command=self._showAssociationsDialog)
        self.ssMenu = Tkinter.Menu(self.structureMenu)
        self.structureMenu.add_cascade(label="Secondary Structure",
                            menu=self.ssMenu)
        self.showSSVar = Tkinter.IntVar(parent)
        self.showSSVar.set(False)
        self.showPredictedSSVar = Tkinter.IntVar(parent)
        self.showPredictedSSVar.set(False)
        self.ssMenu.add_checkbutton(label="show actual",
            variable=self.showSSVar, command=lambda s=self: s.showSS(show=None))
        self.ssMenu.add_checkbutton(label="show predicted",
            variable=self.showPredictedSSVar,
            command=lambda s=self: s.showSS(show=None, ssType="predicted"))
        # actual SS part of MOD_ASSOC handler...
        self._predSSHandler = self.triggers.addHandler(ADD_SEQS,
            lambda a1, a2, a3, s=self:
            s.showSS(show=None, ssType="predicted"), None)
        """
        from chimerax.atomic import get_triggers
        self._atomic_changes_handler = get_triggers().add_handler("changes", self._atomic_changes_cb)

        """TODO
        self.structureMenu.add_command(state='disabled',
                label="Select by Conservation...",
                command=lambda: self._doByConsCB("Select"))
        self.structureMenu.add_command(state='disabled',
                label="Render by Conservation...",
                command=lambda: self._doByConsCB("Render"))
        self.structureMenu.add_command(label="Expand Selection to"
                " Columns", state=assocState,
                command=self.expandSelectionByColumns)
        self._modAssocHandlerID = self.triggers.addHandler(
                    MOD_ASSOC, self._modAssocCB, None)

        self.headersMenu = Tkinter.Menu(self.menuBar)
        self.menuBar.add_cascade(label="Headers", menu=self.headersMenu)
        self.headersMenu.add_command(label="Save...",
            command=self._showSaveHeaderDialog)
        self.headersMenu.add_command(label="Load...",
                    command=self._showLoadHeaderDialog)
        self.headersMenu.add_separator()
        for trig in [ADD_HEADERS,DEL_HEADERS,SHOW_HEADERS,HIDE_HEADERS,
                                MOD_ALIGN]:
            self.triggers.addHandler(trig,
                        self._rebuildHeadersMenu, None)
        self._rebuildHeadersMenu()

        self.numberingsMenu = Tkinter.Menu(self.menuBar)
        self.menuBar.add_cascade(label="Numberings",
                        menu=self.numberingsMenu)
        self.showRulerVar = Tkinter.IntVar(self.headersMenu)
        self.showRulerVar.set(
                len(self.seqs) > 1 and self.prefs[SHOW_RULER_AT_STARTUP])
        self.numberingsMenu.add_checkbutton(label="Overall Alignment",
                        selectcolor="black",
                        variable=self.showRulerVar,
                        command=self.setRulerDisplay)
        self.numberingsMenu.add_separator()
        self.numberingsMenu.add_checkbutton(
                    label="Left Sequence",
                    selectcolor="black",
                    variable=self.leftNumberingVar,
                    command=self.setLeftNumberingDisplay)
        self.numberingsMenu.add_checkbutton(
                    label="Right Sequence",
                    selectcolor="black",
                    variable=self.rightNumberingVar,
                    command=self.setRightNumberingDisplay)
        self.numberingsMenu.add_command(
                    label="Adjust Sequence Numberings...",
                    command = self._showNumberingsDialog)

        self.treeMenu = Tkinter.Menu(self.menuBar)
        self.menuBar.add_cascade(label="Tree", menu=self.treeMenu)
        self.treeMenu.add_command(label="Load...",
                    command=self._showTreeDialog)
        self.showTreeVar = Tkinter.IntVar(self.menuBar)
        self.showTreeVar.set(True)
        self.treeMenu.add_checkbutton(label="Show Tree",
            selectcolor="black",
            variable=self.showTreeVar, command=self._showTreeCB,
            state='disabled')
        self.treeMenu.add_separator()
        self.treeMenu.add_command(label="Extract Subalignment",
            state="disabled", command=self.extractSubalignment)

        self.infoMenu = Tkinter.Menu(self.menuBar)
        self.menuBar.add_cascade(label="Info", menu=self.infoMenu)
        if len(self.seqs) == 1:
            state = "disabled"
        else:
            state = "normal"
        self.infoMenu.add_command(label="Percent Identity...",
                state=state, command=self._showIdentityDialog)
        self.infoMenu.add_command(label="Region Browser",
                    command=self.regionBrowser.enter)
        self.infoMenu.add_command(label="Blast Protein...",
                    command=self._showBlastPdbDialog)
        self.infoMenu.add_command(label="UniProt/CDD Annotations...",
                    command=self._showFetchAnnotationsDialog)
        self.preferencesMenu = Tkinter.Menu(self.menuBar)
        self.menuBar.add_cascade(label="Preferences",
                        menu=self.preferencesMenu)

        from chimera.tkgui import aquaMenuBar
        aquaMenuBar(self.menuBar, parent, row = 0, columnspan = 4)

        for tab in self.prefDialog.tabs:
            self.preferencesMenu.add_command(label=tab,
                command=lambda t=tab: [self.prefDialog.enter(),
                self.prefDialog.notebook.selectpage(t)])

        self.status("Mouse drag to create region (replacing current)\n",
            blankAfter=30, followTime=40, followWith=
            "Shift-drag to add to current region\n"
            "Control-drag to add new region")
        self._addHandlerID = chimera.openModels.addAddHandler(
                        self._newModelsCB, None)
        self._removeHandlerID = chimera.openModels.addRemoveHandler(
                        self._closeModelsCB, None)
        self._closeSessionHandlerID = chimera.triggers.addHandler(
            CLOSE_SESSION, lambda t, a1, a2, s=self: s.Quit(), None)
        self._monitorChangesHandlerID = None
        # deregister other handlers on APPQUIT...
        chimera.triggers.addHandler(chimera.APPQUIT, self.destroy, None)
        if self.autoAssociate == None:
            if len(self.seqs) == 1:
                self.intrinsicStructure = True
            else:
                self.autoAssociate = False
                self.associate(None)
        else:
            self._newModelsCB(models=chimera.openModels.list())
        self._makeSequenceRegions()
        if self.prefs[LOAD_PDB_AUTO]:
            # delay calling _loadStructures to give any structures
            # opened along with MAV a chance to load
            parent.after_idle(lambda: self._loadStructures(auto=1))
        """
        from Qt.QtCore import Qt
        self.tool_window.manage('side', allowed_areas=Qt.DockWidgetArea.AllDockWidgetAreas)

    @property
    def active_region(self):
        return self.region_manager.cur_region()

    def alignment_notification(self, note_name, note_data):
        alignment = self.alignment
        if note_name == alignment.NOTE_MOD_ASSOC:
            assoc_aseqs = set()
            if note_data[0] != alignment.NOTE_DEL_ASSOC:
                match_maps = note_data[1]
            else:
                match_maps = [note_data[1]['match map']]
            for match_map in match_maps:
                aseq = match_map.align_seq
                assoc_aseqs.add(aseq)
            for aseq in assoc_aseqs:
                self.seq_canvas.assoc_mod(aseq)
                self._update_errors_gaps(aseq)
            if self.alignment.intrinsic:
                self.show_ss(True)
            for opt_tool in ['associations_tool', 'transfer_seq_tool']:
                if hasattr(self, opt_tool):
                    getattr(self, opt_tool)._assoc_mod(note_data)
        elif note_name == alignment.NOTE_PRE_DEL_SEQS:
            self.region_manager._pre_remove_lines(note_data)
            for seq in note_data:
                if seq in self._feature_browsers:
                    self._feature_browsers[seq].tool_window.destroy()
                    del self._feature_browsers[seq]
        elif note_name == alignment.NOTE_DESTROYED:
            self.delete()
        elif note_name == alignment.NOTE_COMMAND:
            from .cmd import run
            run(self.session, self, note_data)
        elif note_name == alignment.NOTE_RMSD_UPDATE:
            if self._regions_tool:
                self._regions_tool.alignment_rmsd_update()
        elif note_name == alignment.NOTE_ADD_DEL_SEQS:
            if self._associations_tool:
                self._associations_tool._choose_widgets()

        self.seq_canvas.alignment_notification(note_name, note_data)

    @property
    def consensus_capitalize_theshold(self):
        return self.seq_canvas.consensus_capitalize_theshold

    @consensus_capitalize_theshold.setter
    def consensus_capitalize_theshold(self, capitalize_theshold):
        self.seq_canvas.consensus_capitalize_theshold = capitalize_theshold

    @property
    def consensus_ignores_gaps(self):
        return self.seq_canvas.consensus_ignores_gaps

    @consensus_ignores_gaps.setter
    def consensus_ignores_gaps(self, ignores_gaps):
        self.seq_canvas.consensus_ignores_gaps = ignores_gaps

    @property
    def conservation_style(self):
        return self.seq_canvas.conservation_style

    @conservation_style.setter
    def conservation_style(self, style):
        self.seq_canvas.conservation_style = style

    def delete(self):
        self.region_manager.destroy()
        self.seq_canvas.destroy()
        self.alignment.detach_viewer(self)
        for seq in self.alignment.seqs:
            seq.triggers.remove_handler(self._seq_rename_handlers[seq])
        from chimerax.atomic import get_triggers
        get_triggers().remove_handler(self._atomic_changes_handler)
        ToolInstance.delete(self)

    def expand_selection_to_columns(self):
        from chimerax.core.commands import StringArg, run
        if len(self.session.alignments) > 1:
            align_arg = ' ' + StringArg.unparse(self.alignment.ident)
        else:
            align_arg = ''
        run(self.session, "seq expandsel" + align_arg)

    def fill_context_menu(self, menu, x, y):
        from Qt.QtGui import QAction
        file_menu = menu.addMenu("File")
        save_as_menu = file_menu.addMenu("Save As")
        from chimerax.core.commands import run, StringArg
        align_arg = "%s " % self.alignment if len(self.session.alignments.alignments) > 1 else ""
        fmts = [fmt for fmt in self.session.save_command.save_data_formats if fmt.category == "Sequence"]
        fmts.sort(key=lambda fmt: fmt.synopsis.casefold())
        for fmt in fmts:
            action = QAction(fmt.synopsis, save_as_menu)
            action.triggered.connect(lambda *args, fmt=fmt:
                run(self.session, "save browse format %s alignment %s"
                % (fmt.nicknames[0], StringArg.unparse(self.alignment.ident))))
            save_as_menu.addAction(action)
        scf_action = QAction("Load Sequence Coloring File...", file_menu)
        scf_action.triggered.connect(lambda: self.load_scf_file(None))
        file_menu.addAction(scf_action)

        edit_menu = menu.addMenu("Edit")
        copy_action = QAction("Copy Sequence...", edit_menu)
        copy_action.triggered.connect(self.show_copy_sequence_dialog)
        edit_menu.addAction(copy_action)
        single_seq = len(self.alignment.seqs) == 1
        from chimerax.seqalign.cmd import alignment_program_name_args
        prog_to_arg = {}
        for arg, prog in alignment_program_name_args.items():
            prog_to_arg[prog] = arg
        for prog in sorted(prog_to_arg.keys()):
            prog_menu = edit_menu.addMenu("Realign Sequences with %s" % prog)
            for menu_text, cmd_text in [("new", ""), ("this", " replace true")]:
                realign_action = QAction("in %s window" % menu_text, prog_menu)
                realign_action.triggered.connect(lambda *args, arg=prog_to_arg[prog],
                    unparse=StringArg.unparse, cmd_text=cmd_text: run(self.session,
                    "seq align %s program %s%s" % (unparse(self.alignment.ident), unparse(arg), cmd_text)))
                prog_menu.addAction(realign_action)
            if single_seq:
                prog_menu.setEnabled(False)
        rename_action = QAction("Rename Sequence...", edit_menu)
        rename_action.triggered.connect(self.show_rename_sequence_dialog)
        edit_menu.addAction(rename_action)

        structure_menu = menu.addMenu("Structure")
        assoc_action = QAction("Associations...", structure_menu)
        assoc_action.triggered.connect(self.show_associations)
        from chimerax.atomic import AtomicStructure
        for m in self.session.models:
            if isinstance(m, AtomicStructure):
                break
        else:
            assoc_action.setEnabled(False)
        structure_menu.addAction(assoc_action)
        if len(self.alignment.seqs) > 1:
            expand_action = QAction("Expand Selection to Columns", structure_menu)
            expand_action.triggered.connect(self.expand_selection_to_columns)
            expand_action.setEnabled(bool(self.alignment.associations))
            structure_menu.addAction(expand_action)
            cons_sel_menu = structure_menu.addMenu("Select by Column Identity")
            cons_sel_menu.setEnabled(bool(self.alignment.associations))
            for entry_text, value in [("100%", 100.0), ("<100%", -100.0), (">=50%", 50.0), ("<50%", -50.0)]:
                action = QAction(entry_text, cons_sel_menu)
                action.triggered.connect(lambda act, *args, func=self.select_by_column_identity, val=value:
                    func(val))
                cons_sel_menu.addAction(action)
        match_action = QAction("Match...", structure_menu)
        match_action.triggered.connect(self.show_match_dialog)
        match_action.setEnabled(
            len(set([chain.structure for chain in self.alignment.associations.keys()])) > 1)
        structure_menu.addAction(match_action)
        xfer_action = QAction("Update Chain Sequence...", structure_menu)
        xfer_action.triggered.connect(self.show_transfer_seq_dialog)
        xfer_action.setEnabled(bool(self.alignment.associations))
        structure_menu.addAction(xfer_action)
        view_targets = []
        # bounded_by expects the scene coordinate system...
        from Qt.QtCore import QPointF
        global_xy = self.tool_window.ui_area.mapToGlobal(QPointF(x, y))
        view_xy = self.seq_canvas.main_view.mapFromGlobal(global_xy)
        scene_xy = self.seq_canvas.main_view.mapToScene(view_xy.toPoint())
        x, y = scene_xy.x(), scene_xy.y()

        seq, seq, index, index = self.seq_canvas.bounded_by(x, y, x, y, exclude_headers=True)
        if seq is not None and seq in self.alignment.seqs:
            for chain, mm in self.alignment.match_maps[seq].items():
                try:
                    view_targets.append(mm[seq.gapped_to_ungapped(index)])
                except KeyError:
                    continue
        if view_targets:
            if len(view_targets) == 1:
                view_action = QAction("View %s" % view_targets[0], structure_menu)
                view_action.triggered.connect(lambda *args, cmd_text="view %s" % view_targets[0].atomspec:
                    run(self.session, cmd_text))
                structure_menu.addAction(view_action)
            else:
                view_menu = structure_menu.addMenu("View Residue")
                view_targets.sort()
                specs = [vt.atomspec for vt in view_targets]
                for target, spec in zip(view_targets, specs):
                    view_action = QAction(str(target), view_menu)
                    view_action.triggered.connect(lambda *args, cmd_text="view %s" % spec:
                        run(self.session, cmd_text))
                    view_menu.addAction(view_action)
                view_menu.addSeparator()
                view_action = QAction("All The Above", view_menu)
                view_action.triggered.connect(lambda *args, cmd_text="view %s" % ' '.join(specs):
                    run(self.session, cmd_text))
                view_menu.addAction(view_action)
        else:
            view_action = QAction("View Residue", structure_menu)
            view_action.setEnabled(False)
            structure_menu.addAction(view_action)

        self.alignment.add_headers_menu_entry(menu)

        numberings_menu = menu.addMenu("Numberings")
        action = QAction("Overall", numberings_menu)
        action.setCheckable(True)
        action.setChecked(self.seq_canvas.show_ruler)
        action.triggered.connect(lambda*, sc=self.seq_canvas, action=action:
            setattr(sc, "show_ruler", action.isChecked()))
        numberings_menu.addAction(action)
        refseq_menu = numberings_menu.addMenu("Reference Sequence")
        action = QAction("No Reference Sequence", refseq_menu)
        action.setCheckable(True)
        action.setChecked(self.alignment.reference_seq is None)
        action.triggered.connect(lambda*, align_arg=align_arg[:-1], action=action, self=self:
            run(self.session, "seq ref " + align_arg) if action.isChecked() else None)
        refseq_menu.addAction(action)
        for seq in self.alignment.seqs:
            action = QAction(seq.name, refseq_menu)
            action.setCheckable(True)
            action.setChecked(self.alignment.reference_seq is seq)
            action.triggered.connect(lambda*, seq_arg=StringArg.unparse(align_arg[:-1] + ':' + seq.name),
                action=action: run(self.session, "seq ref " + seq_arg) if action.isChecked() else None)
            refseq_menu.addAction(action)
        numberings_menu.addSeparator()
        action = QAction("Left Sequence", numberings_menu)
        action.setCheckable(True)
        action.setChecked(self.seq_canvas.show_left_numbering)
        action.triggered.connect(lambda*, sc=self.seq_canvas, action=action:
            setattr(sc, "show_left_numbering", action.isChecked()))
        numberings_menu.addAction(action)
        action = QAction("Right Sequence", numberings_menu)
        action.setCheckable(True)
        action.setChecked(self.seq_canvas.show_right_numbering)
        action.triggered.connect(lambda*, sc=self.seq_canvas, action=action:
            setattr(sc, "show_right_numbering", action.isChecked()))
        numberings_menu.addAction(action)

        tools_menu = menu.addMenu("Tools")
        comp_model_action = QAction("Modeller Comparative Modeling...", tools_menu)
        comp_model_action.triggered.connect(lambda: run(self.session,
            "ui tool show 'Modeller Comparative'"))
        if not self.alignment.associations:
            comp_model_action.setEnabled(False)
        tools_menu.addAction(comp_model_action)
        loops_model_action = QAction("Model Loops...", tools_menu)
        loops_model_action.triggered.connect(lambda: run(self.session,
            "ui tool show 'Model Loops'"))
        if not self.alignment.associations:
            loops_model_action.setEnabled(False)
        tools_menu.addAction(loops_model_action)
        if len(self.alignment.seqs) == 1:
            from chimerax.blastprotein import BlastProteinTool
            blast_action = QAction("Blast Protein...", tools_menu)
            blast_action.triggered.connect(
                lambda: BlastProteinTool(self.session, sequences = StringArg.unparse("%s:1" % self.alignment.ident))
            )
            tools_menu.addAction(blast_action)
        else:
            from chimerax.blastprotein import BlastProteinTool
            blast_menu = tools_menu.addMenu("Blast Protein")
            for i, seq in enumerate(self.alignment.seqs):
                blast_action = QAction(seq.name, blast_menu)
                blast_action.triggered.connect(lambda *args, chars=seq.ungapped():
                    BlastProteinTool(self.session, sequences=StringArg.unparse(chars)))
                blast_menu.addAction(blast_action)
        if len(self.alignment.seqs) > 1:
            identity_action = QAction("Percent Identity...", menu)
            identity_action.triggered.connect(self.show_percent_identity_dialog)
            tools_menu.addAction(identity_action)

        annotations_menu = menu.addMenu("Annotations")
        rt_action = QAction("Regions...", annotations_menu)
        rt_action.triggered.connect(lambda*, self=self, action=rt_action:
            setattr(self, "regions_tool_shown", True))
        annotations_menu.addAction(rt_action)
        feature_seqs = [ seq for seq in self.alignment.seqs if seq.features(fetch=False) ]
        if feature_seqs:
            if len(self.alignment.seqs) == 1:
                action = QAction("Sequence Features...", annotations_menu)
                action.triggered.connect(lambda *args, seq=feature_seqs[0], show=self.show_feature_browser:
                    show(seq))
                annotations_menu.addAction(action)
            else:
                features_menu = annotations_menu.addMenu("Sequence Features")
                from .seq_canvas import _seq_name as seq_name
                for seq in feature_seqs:
                    action = QAction(seq_name(seq), features_menu)
                    action.triggered.connect(lambda *args, seq=seq, show=self.show_feature_browser:
                        show(seq))
                    features_menu.addAction(action)

        settings_action = QAction("Settings...", menu)
        settings_action.triggered.connect(self.show_settings)
        menu.addAction(settings_action)

    def load_scf_file(self, path, color_structures=None):
        """color_structures=None means use user's preference setting"""
        self.region_manager.load_scf_file(path, color_structures)

    def new_region(self, name=None, **kw):
        if 'blocks' in kw:
            # interpret numeric values as indices into sequences
            blocks = kw['blocks']
            if blocks and isinstance(blocks[0][0], int):
                blocks = [(self.alignment.seqs[i1], self.alignment.seqs[i2], i3, i4)
                        for i1, i2, i3, i4 in blocks]
                kw['blocks'] = blocks
        if 'columns' in kw:
            # in lieu of specifying blocks, allow list of columns
            # (implicitly all rows); list should already be in order
            left = right = None
            blocks = []
            for col in kw['columns']:
                if left is None:
                    left = right = col
                elif col > right + 1:
                    blocks.append((self.alignment.seqs[0], self.alignment.seqs[-1], left, right))
                    left = right = col
                else:
                    right = col
            if left is not None:
                blocks.append((self.alignment.seqs[0], self.alignment.seqs[-1], left, right))
            kw['blocks'] = blocks
            del kw['columns']
        return self.region_manager.new_region(name, **kw)

    @property
    def regions_tool_shown(self):
        return self._regions_tool is not None and self._regions_tool.shown

    @regions_tool_shown.setter
    def regions_tool_shown(self, shown):
        if self._regions_tool is None:
            from .region_browser import RegionsTool
            rt_window = self.tool_window.create_child_window("Regions", close_destroys=False, statusbar=True)
            self._regions_tool = RegionsTool(self, rt_window)
            rt_window.fill_context_menu = self.fill_context_menu
            rt_window.manage(None)
        self._regions_tool.shown = shown

    def select_by_column_identity(self, col_percent):
        """Select residues associated with columns that match the criteria, which is a percentage value.
           Positive values mean greater than or equal to, and negative values mean less than.
        """
        from chimerax.core.commands import StringArg, run
        log = len(self.session.alignments) > 1
        run(self.session, "seq refreshAttrs " + StringArg.unparse(self.alignment.ident), log=log)
        if col_percent < 0.0:
            criteria = "<%g" % -col_percent
        else:
            criteria = ">=%g" % col_percent
        from chimerax.atomic import concise_residue_spec
        spec = concise_residue_spec(self.session, self.alignment.associated_residues())
        run(self.session, "select ::%s%s & %s" % (self.alignment.COL_IDENTITY_ATTR, criteria, spec))

    def show_associations(self):
        if not hasattr(self, "associations_tool"):
            from .associations_tool import AssociationsTool
            self.associations_tool = AssociationsTool(self,
                self.tool_window.create_child_window("Chain-Sequence Associations", close_destroys=False))
            self.associations_tool.tool_window.manage(None)
        self.associations_tool.tool_window.shown = True

    def show_copy_sequence_dialog(self):
        if not hasattr(self, "copy_sequence_dialog"):
            from .copy_seq import CopySeqDialog
            self.copy_sequence_dialog = CopySeqDialog(self,
                self.tool_window.create_child_window("Copy Sequence", close_destroys=False))
            self.copy_sequence_dialog.tool_window.manage(None)
        self.copy_sequence_dialog.tool_window.shown = True

    def show_feature_browser(self, seq, *, state=None):
        if seq not in self._feature_browsers:
            from .feature_browser import FeatureBrowser
            self._feature_browsers[seq] = FeatureBrowser(self, seq, state,
                self.tool_window.create_child_window("%s Features" % seq.name, close_destroys=False))
            self._feature_browsers[seq].tool_window.manage(None)
        self._feature_browsers[seq].tool_window.shown = True

    def show_match_dialog(self):
        if not hasattr(self, "match_dialog"):
            from .match import MatchDialog
            self.match_dialog = MatchDialog(self,
                self.tool_window.create_child_window("Match", close_destroys=False))
            self.match_dialog.tool_window.manage(None)
        self.match_dialog.tool_window.shown = True

    def show_percent_identity_dialog(self):
        if not hasattr(self, "percent_identity_dialog"):
            from .identity import PercentIdentityDialog
            self.percent_identity_dialog = PercentIdentityDialog(self,
                self.tool_window.create_child_window("Percent Identity", close_destroys=False))
            self.percent_identity_dialog.tool_window.manage(None)
        self.percent_identity_dialog.tool_window.shown = True

    def show_rename_sequence_dialog(self):
        if not hasattr(self, "rename_sequence_dialog"):
            from .rename_seq import RenameSeqDialog
            self.rename_sequence_dialog = RenameSeqDialog(self,
                self.tool_window.create_child_window("Rename Sequence", close_destroys=False))
            self.rename_sequence_dialog.tool_window.manage(None)
        self.rename_sequence_dialog.tool_window.shown = True

    def show_transfer_seq_dialog(self):
        if not hasattr(self, "transfer_seq_dialog"):
            from .transfer_seq_tool import TransferSeqTool
            self.transfer_seq_dialog = TransferSeqTool(self,
                self.tool_window.create_child_window("Update Chain Sequence", close_destroys=False))
            self.transfer_seq_dialog.tool_window.manage(None)
        self.transfer_seq_dialog.tool_window.shown = True

    def show_settings(self):
        if not hasattr(self, "settings_tool"):
            from .settings_tool import SettingsTool
            self.settings_tool = SettingsTool(self,
                self.tool_window.create_child_window("Sequence Viewer Settings", close_destroys=False))
            self.settings_tool.tool_window.manage(None)
        self.settings_tool.tool_window.shown = True

    def show_ss(self, show=True):
        # show == None means don't change show states, but update regions
        # ... not yet implemented, so see if the regions exist and their
        # display is True...
        rm = self.region_manager
        if show == None:
            hreg = rm.get_region(rm.ACTUAL_HELICES_REG_NAME)
            if not hreg:
                return
            show = hreg.shown
        rm.show_ss(show)

    def status(self, *args, **kw):
        status = self.tool_window.status(*args, **kw)
        if self._regions_tool:
            self._regions_tool.tool_window.status(*args, **kw)

    @classmethod
    def restore_snapshot(cls, session, data):
        inst = super().restore_snapshot(session, data['ToolInstance'])
        inst._finalize_init(data['alignment'])
        if 'seq canvas' in data:
            inst.seq_canvas.restore_state(session, data['seq canvas'])
        inst.region_manager.restore_state(data['region browser'])
        # feature browsers depend on regions (and therefore the region browser) being restored first
        if 'feature browsers' in data:
            from .feature_browser import FeatureBrowser
            for seq, fb_data in data['feature browsers'].items():
                inst.show_feature_browser(seq, state=fb_data)
        return inst

    SESSION_SAVE = True

    def take_snapshot(self, session, flags):
        data = {
            'ToolInstance': ToolInstance.take_snapshot(self, session, flags),
            'alignment': self.alignment,
            'feature browsers': {seq: fb.state() for seq, fb in self._feature_browsers.items()},
            'region browser': self.region_manager.state(),
            'seq canvas': self.seq_canvas.state()
        }
        return data

    def _atomic_changes_cb(self, trig_name, changes):
        if "ss_type changed" in changes.residue_reasons():
            self.show_ss(show=None)
        if self._regions_tool:
            self._regions_tool._atomic_changes_cb(changes)

    def _regions_tool_notification(self, category, region):
        if self._regions_tool:
            self._regions_tool.region_notification(category, region)
        try:
            fb = self._feature_browsers[region.sequence]
        except KeyError:
            pass
        else:
            fb.region_notification(category, region)

    def _update_errors_gaps(self, aseq):
        if not self.settings.error_region_shown and not self.settings.gap_region_shown:
            return
        a_ref_seq = getattr(aseq, 'residue_sequence', aseq.ungapped())
        errors = [0] * len(a_ref_seq)
        gaps = [0] * len(a_ref_seq)
        from chimerax.atomic import Sequence
        for chain, match_map in self.alignment.match_maps[aseq].items():
            for i, char in enumerate(a_ref_seq):
                try:
                    res = match_map[i]
                except KeyError:
                    gaps[i] += 1
                else:
                    if Sequence.rname3to1(res.name) != char.upper():
                        errors[i] += 1
        partial_error_blocks, full_error_blocks = [], []
        partial_gap_blocks, full_gap_blocks = [], []
        num_assocs = len(self.alignment.match_maps[aseq])
        if num_assocs > 0:
            for partial, full, check in [(partial_error_blocks, full_error_blocks, errors),
                    (partial_gap_blocks, full_gap_blocks, gaps)]:
                cur_partial_block = cur_full_block = None
                for i, check_num in enumerate(check):
                    gapped_i = aseq.ungapped_to_gapped(i)
                    if check_num == num_assocs:
                        if cur_full_block:
                            cur_full_block[-1] = gapped_i
                        else:
                            cur_full_block = [aseq, aseq, gapped_i, gapped_i]
                            full.append(cur_full_block)
                        if cur_partial_block:
                            cur_partial_block = None
                    else:
                        if cur_full_block:
                            cur_full_block = None
                        if check_num > 0:
                            if cur_partial_block:
                                cur_partial_block[-1] = gapped_i
                            else:
                                cur_partial_block = [aseq, aseq, gapped_i, gapped_i]
                                partial.append(cur_partial_block)
                        elif cur_partial_block:
                            cur_partial_block = None

        for shown, region_name_part, partial_blocks, full_blocks, fills, outlines in [
                (self.settings.error_region_shown, self.ERROR_REGION_STRING, partial_error_blocks,
                    full_error_blocks, self.settings.error_region_interiors,
                    self.settings.error_region_borders),
                (self.settings.gap_region_shown, self.GAP_REGION_STRING, partial_gap_blocks,
                    full_gap_blocks, self.settings.gap_region_interiors,
                    self.settings.gap_region_borders)]:
            if not shown:
                continue
            full_fill, partial_fill = fills
            full_outline, partial_outline = outlines
            for region_name_start, blocks, fill, outline in [
                    (region_name_part, full_blocks, full_fill, full_outline),
                    ("partial " + region_name_part, partial_blocks, partial_fill, partial_outline)]:
                region_name = "%s of %s" % (region_name_start, aseq.name)
                old_reg = self.region_manager.get_region(region_name, create=False)
                if old_reg:
                    self.region_manager.delete_region(old_reg)
                if blocks:
                    self.region_manager.new_region(region_name, blocks=blocks, fill=fill,
                        outline=outline, sequence=aseq, cover_gaps=False)

def _start_seq_viewer(session, tool_name, alignment):
    return SequenceViewer(session, tool_name, alignment)

