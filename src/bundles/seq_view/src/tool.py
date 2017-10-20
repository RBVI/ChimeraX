# vim: set expandtab ts=4 sw=4:

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

from chimerax.core.tools import ToolInstance
class SequenceViewer(ToolInstance):
    """ Viewer displays a multiple sequence alignment """

    MATCHED_REGION_INFO = ("matched residues", (1, .88, .8), "orange red")

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
            set_state_from_snapshot will be called later.

            if 'autoAssociate' is None then it is the same as False except
            that any StructureSequences in the alignment will be associated
            with their structures
        """

        ToolInstance.__init__(self, session, tool_name)
        if alignment is None:
            return
        self._finalize_init(session, alignment)

    def _finalize_init(self, session, alignment):
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
        from . import settings
        self.settings = settings.init(session)
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
        words = self.alignment.description.split()
        capped_words = []
        for word in words:
            if word.islower() and word.isalpha():
                capped_words.append(word.capitalize())
            else:
                capped_words.append(word)
        self.display_name = " ".join(capped_words) + " [ID: %s]" % self.alignment.ident
        from chimerax.core.ui.gui  import MainToolWindow
        self.tool_window = MainToolWindow(self, close_destroys=True, statusbar=True)
        self.tool_window._ToolWindow__toolkit.dock_widget.setMouseTracking(True)
        self.status = self.tool_window.status
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
        from .region_browser import RegionBrowser
        rb_window = self.tool_window.create_child_window("Regions", close_destroys=False)
        self.region_browser = RegionBrowser(rb_window, self.seq_canvas)
        self._seq_rename_handlers = {}
        for seq in self.alignment.seqs:
            self._seq_rename_handlers[seq] = seq.triggers.add_handler("rename",
                self.region_browser._seq_renamed_cb)
            if seq.match_maps:
               self._update_errors_gaps(seq)
        if self.alignment.intrinsic:
            self.show_ss(True)
            self.status("Helices/strands depicted in gold/green")
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
        from chimerax.core.atomic import get_triggers
        self._atomic_changes_handler = get_triggers(self.session).add_handler(
            "changes", self._atomic_changes_cb)

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
        self.tool_window.manage('side' if self.seq_canvas.wrap_okay() else 'top')

    def alignment_notification(self, note_name, note_data):
        if note_name == "modify association":
            assoc_aseqs = set()
            for match_map in note_data[-1]:
                aseq = match_map.align_seq
                assoc_aseqs.add(aseq)
            for aseq in assoc_aseqs:
                self.seq_canvas.assoc_mod(aseq)
                self._update_errors_gaps(aseq)
        elif note_name == "pre-remove seqs":
            self.region_browser._pre_remove_lines(note_data)
        elif note_name == "destroyed":
            self.delete()

    def delete(self):
        self.region_browser.destroy()
        self.seq_canvas.destroy()
        self.alignment.detach_viewer(self)
        for seq in self.alignment.seqs:
            seq.triggers.remove_handler(self._seq_rename_handlers[seq])
        from chimerax.core.atomic import get_triggers
        get_triggers(self.session).remove_handler(self._atomic_changes_handler)
        ToolInstance.delete(self)

    def new_region(self, **kw):
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
        return self.region_browser.new_region(**kw)

    def show_ss(self, show=True):
        # show == None means don't change show states, but update regions
        # ... not yet implemented, so see if the regions exist and their
        # display is True...
        rb = self.region_browser
        if show == None:
            hreg = rb.get_region(rb.ACTUAL_HELICES_REG_NAME)
            if not hreg:
                return
            show = hreg.shown
        rb.show_ss(show)

    @classmethod
    def restore_snapshot(cls, session, data):
        bundle_info = session.toolshed.find_bundle_for_class(cls)
        inst = cls(session, bundle_info.tools[0].name)
        ToolInstance.set_state_from_snapshot(inst, session, data['ToolInstance'])
        inst._finalize_init(session, data['alignment'])
        inst.region_browser.restore_state(data['region browser'])
        return inst

    SESSION_SAVE = True
    
    def take_snapshot(self, session, flags):
        data = {
            'ToolInstance': ToolInstance.take_snapshot(self, session, flags),
            'alignment': self.alignment,
            'region browser': self.region_browser.save_state()
        }
        return data

    def _atomic_changes_cb(self, trig_name, changes):
        if "ss_type changed" in changes.residue_reasons():
            self.show_ss(show=None)

    def _update_errors_gaps(self, aseq):
        if not self.settings.error_region_shown and not self.settings.gap_region_shown:
            return
        a_ref_seq = getattr(aseq, 'residue_sequence', aseq.ungapped())
        errors = [0] * len(a_ref_seq)
        gaps = [0] * len(a_ref_seq)
        from chimerax.core.atomic import Sequence
        for chain, match_map in aseq.match_maps.items():
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
        num_assocs = len(aseq.match_maps)
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
                (self.settings.error_region_shown, "mismatches", partial_error_blocks,
                    full_error_blocks, self.settings.error_region_interiors,
                    self.settings.error_region_borders),
                (self.settings.gap_region_shown, "missing structure", partial_gap_blocks,
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
                old_reg = self.region_browser.get_region(region_name, create=False)
                if old_reg:
                    self.region_browser.delete_region(old_reg)
                if blocks:
                    self.region_browser.new_region(region_name, blocks=blocks, fill=fill,
                        outline=outline, sequence=aseq, cover_gaps=False)

def _start_seq_viewer(session, tool_name, alignment=None):
    if alignment is None:
        from chimerax.core.errors import LimitationError
        raise LimitationError("Running MAV from tools menu not implemented; instead, open"
            " alignment using 'open' command or File->Open")
    return SequenceViewer(session, tool_name, alignment)

