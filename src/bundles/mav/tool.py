# vim: set expandtab ts=4 sw=4:

from chimerax.core.tools import ToolInstance
class MultalignViewer(ToolInstance):
    """ Viewer displays a multiple sequence alignment """

    """TODO
    buttons = ('Quit', 'Hide')
    help = "ContributedSoftware/multalignviewer/framemav.html"
    provideStatus = True
    statusWidth = 15
    statusPosition = "left"
    provideSecondaryStatus = True
    secondaryStatusPosition = "left"

    ConsAttr = "mavPercentConserved"

    MATCH_REG_NAME_START = "matches"
    ERROR_REG_NAME_START = "mismatches"
    GAP_REG_NAME_START = "missing structure"

    # so Model Loops tool can invoke it...
    MODEL_LOOPS_MENU_TEXT = "Modeller (loops/refinement)..."
    import RegionBrowser
    SEL_REGION_NAME = RegionBrowser.SEL_REGION_NAME

    def __init__(self, fileNameOrSeqs, fileType=None, autoAssociate=True,
                title=None, quitCB=None, frame=None, numberingDisplay=None,
                sessionSave=True):
    """
    def __init__(self, session, tool_name, alignment):
        """ if 'autoAssocate' is None then it is the same as False except
            that any StructureSequences in the alignment will be associated
            with their structures
        """
        ToolInstance.__init__(self, session, tool_name)

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
        alignment.attach_viewer(self)
        """
        self.prefs = prefs
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

        self._seqRenameHandlers = {}
        for seq in seqs:
            self._seqRenameHandlers[seq] = seq.triggers.addHandler(
                seq.TRIG_RENAME, self._seqRenameCB, None)
        self._defaultNumbering = defaultNumbering
        self.fileAttrs = fileAttrs
        self.fileMarkups = fileMarkups
        if not title:
            if isinstance(fileNameOrSeqs, basestring):
                title = os.path.split(fileNameOrSeqs)[1]
            else:
                title = "MultAlignViewer"
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
        from chimerax.core.ui.gui  import MainToolWindow
        self.tool_window = MainToolWindow(self)
        self.tool_window._ToolWindow__toolkit.dock_widget.setMouseTracking(True)
        self.tool_window.close_destroys = True
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
                if aseq.match_maps:
                    self.seq_canvas.assoc_mod(aseq)
        """TODO
        self.regionBrowser = RegionBrowser(self.seq_canvas)
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
        self._resChangeHandler = chimera.triggers.addHandler(
            "Residue", self._resChangeCB, None)

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
        self.tool_window.manage('side' if self.seq_canvas.should_wrap() else 'top')

    def alignment_notification(self, note_name, note_data):
        if note_name == "mod assoc":
            for match_map in note_data[-1]:
                self.seq_canvas.assoc_mod(match_map.aseq)

    def delete(self):
        self.alignment.detach_viewer(self)
        ToolInstance.delete(self)

def _start_mav(session, tool_name, alignment=None):
    if alignment is None:
        raise LimitationError("Running MAV from tools menu not implemented; instead, open"
            " alignment using 'open' command or File->Open")
    return MultalignViewer(session, tool_name, alignment)
