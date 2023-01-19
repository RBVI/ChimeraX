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
class ProfileGridsTool(ToolInstance):
    """ Viewer displays a multiple sequence alignment as a grid/table """

    #help = "help:user/tools/sequenceviewer.html"
    help = None

    def __init__(self, session, tool_name, alignment=None):
        """ if 'alignment' is None, then we are being restored from a session and
            _finalize_init will be called later.
        """

        ToolInstance.__init__(self, session, tool_name)
        if alignment is None:
            return
        self._finalize_init(alignment)

    def _finalize_init(self, alignment, *, session_data=None):
        self.alignment = alignment
        from . import subcommand_name
        alignment.attach_viewer(self, subcommand_name=subcommand_name)
        from . import settings
        self.settings = settings.init(self.session)
        from chimerax.core.utils import titleize
        self.display_name = titleize(self.alignment.description) + " [ID: %s]" % self.alignment.ident
        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self, close_destroys=True, statusbar=True)
        self.tool_window._dock_widget.setMouseTracking(True)
        #self.tool_window.fill_context_menu = self.fill_context_menu
        self.status = self.tool_window.status
        parent = self.tool_window.ui_area
        parent.setMouseTracking(True)
        from .grid_canvas import GridCanvas
        if session_data is None:
            import os
            num_cpus = os.cpu_count()
            if num_cpus is None:
                num_cpus = 1
            from ._profile_grids import compute_profile
            weights = [getattr(seq, 'weight', 1.0) for seq in alignment.seqs]
            grid_data = compute_profile([seq.cpp_pointer for seq in alignment.seqs], weights, num_cpus)
        else:
            grid_data, weights = session_data
        self.grid_canvas = GridCanvas(parent, self, self.alignment, grid_data, weights)
        #self._seq_rename_handlers = {}
        #for seq in self.alignment.seqs:
        #    self._seq_rename_handlers[seq] = seq.triggers.add_handler("rename",
        #        self.region_browser._seq_renamed_cb)

        self.tool_window.manage('side')


    def alignment_notification(self, note_name, note_data):
        raise NotImplementedError("alignment_notification")
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
                self.grid_canvas.assoc_mod(aseq)
                self._update_errors_gaps(aseq)
            if self.alignment.intrinsic:
                self.show_ss(True)
            if hasattr(self, 'associations_tool'):
                self.associations_tool._assoc_mod(note_data)
        elif note_name == alignment.NOTE_PRE_DEL_SEQS:
            self.region_browser._pre_remove_lines(note_data)
            for seq in note_data:
                if seq in self._feature_browsers:
                    self._feature_browsers[seq].tool_window.destroy()
                    del self._feature_browsers[seq]
        elif note_name == alignment.NOTE_DESTROYED:
            self.delete()
        elif note_name == alignment.NOTE_COMMAND:
            from .cmd import run
            run(self.session, self, note_data)

        self.grid_canvas.alignment_notification(note_name, note_data)

    def delete(self):
        self.grid_canvas.destroy()
        self.alignment.detach_viewer(self)
        for seq in self.alignment.seqs:
            seq.triggers.remove_handler(self._seq_rename_handlers[seq])
        ToolInstance.delete(self)

    def fill_context_menu(self, menu, x, y):
        raise NotImplementedError("fill_context_menu")
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

        headers_menu = menu.addMenu("Headers")
        headers = self.alignment.headers
        headers.sort(key=lambda hdr: hdr.ident.casefold())
        from chimerax.core.commands import run
        for hdr in headers:
            action = QAction(hdr.name, headers_menu)
            action.setCheckable(True)
            action.setChecked(hdr.shown)
            if not hdr.relevant:
                action.setEnabled(False)
            action.triggered.connect(lambda *, action=action, hdr=hdr, align_arg=align_arg, self=self: run(
                self.session, "seq header %s%s %s" % (align_arg, hdr.ident, "show" if action.isChecked() else "hide")))
            headers_menu.addAction(action)
        headers_menu.addSeparator()
        hdr_save_menu = headers_menu.addMenu("Save")
        for hdr in headers:
            if not hdr.relevant:
                continue
            action = QAction(hdr.name, hdr_save_menu)
            action.triggered.connect(lambda *, hdr=hdr, align_arg=align_arg, self=self: run(
                self.session, "seq header %s%s save browse" % (align_arg, hdr.ident)))
            hdr_save_menu.addAction(action)

        numberings_menu = menu.addMenu("Numberings")
        action = QAction("Overall", numberings_menu)
        action.setCheckable(True)
        action.setChecked(self.grid_canvas.show_ruler)
        action.triggered.connect(lambda*, sc=self.grid_canvas, action=action:
            setattr(sc, "show_ruler", action.isChecked()))
        numberings_menu.addAction(action)
        refseq_menu = numberings_menu.addMenu("Reference Sequence")
        action = QAction("No Reference Sequence", refseq_menu)
        action.setCheckable(True)
        action.setChecked(self.alignment.reference_seq is None)
        action.triggered.connect(lambda*, align_arg=align_arg, action=action, self=self:
            run(self.session, "seq ref " + align_arg) if action.isChecked() else None)
        refseq_menu.addAction(action)
        for seq in self.alignment.seqs:
            action = QAction(seq.name, refseq_menu)
            action.setCheckable(True)
            action.setChecked(self.alignment.reference_seq is seq)
            action.triggered.connect(lambda*, seq_arg=StringArg.unparse(align_arg + ':' + seq.name),
                action=action: run(self.session, "seq ref " + seq_arg) if action.isChecked() else None)
            refseq_menu.addAction(action)
        numberings_menu.addSeparator()
        action = QAction("Left Sequence", numberings_menu)
        action.setCheckable(True)
        action.setChecked(self.grid_canvas.show_left_numbering)
        action.triggered.connect(lambda*, sc=self.grid_canvas, action=action:
            setattr(sc, "show_left_numbering", action.isChecked()))
        numberings_menu.addAction(action)
        action = QAction("Right Sequence", numberings_menu)
        action.setCheckable(True)
        action.setChecked(self.grid_canvas.show_right_numbering)
        action.triggered.connect(lambda*, sc=self.grid_canvas, action=action:
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


        # Whenever Region Browser and UniProt Annotations happen, the thought is to
        # put them in an "Annotations" menu (rather than "Info"); for now with only
        # sequence features available, use "Features"
        feature_seqs = [ seq for seq in self.alignment.seqs if seq.features(fetch=False) ]
        if feature_seqs:
            if len(self.alignment.seqs) == 1:
                action = QAction("Sequence Features...", menu)
                action.triggered.connect(lambda *args, seq=feature_seqs[0], show=self.show_feature_browser:
                    show(seq))
                menu.addAction(action)
            else:
                features_menu = menu.addMenu("Sequence Features")
                from .grid_canvas import _seq_name as seq_name
                for seq in feature_seqs:
                    action = QAction(seq_name(seq), features_menu)
                    action.triggered.connect(lambda *args, seq=seq, show=self.show_feature_browser:
                        show(seq))
                    features_menu.addAction(action)

        settings_action = QAction("Settings...", menu)
        settings_action.triggered.connect(self.show_settings)
        menu.addAction(settings_action)


    def show_settings(self):
        raise NotImplementedError("show_settings")
        if not hasattr(self, "settings_tool"):
            from .settings_tool import SettingsTool
            self.settings_tool = SettingsTool(self,
                self.tool_window.create_child_window("Sequence Viewer Settings", close_destroys=False))
            self.settings_tool.tool_window.manage(None)
        self.settings_tool.tool_window.shown = True

    @classmethod
    def restore_snapshot(cls, session, data):
        raise NotImplementedError("restore_snaphot")
        inst = super().restore_snapshot(session, data['ToolInstance'])
        inst._finalize_init(data['alignment'])
        inst.region_browser.restore_state(data['region browser'])
        if 'seq canvas' in data:
            inst.grid_canvas.restore_state(session, data['seq canvas'])
        # feature browsers depend on regions (and therefore the region browser) being restored first
        if 'feature browsers' in data:
            from .feature_browser import FeatureBrowser
            for seq, fb_data in data['feature browsers'].items():
                inst.show_feature_browser(seq, state=fb_data)
        return inst

    SESSION_SAVE = True

    def take_snapshot(self, session, flags):
        raise NotImplementedError("take_snaphot")
        data = {
            'ToolInstance': ToolInstance.take_snapshot(self, session, flags),
            'alignment': self.alignment,
            'feature browsers': {seq: fb.state() for seq, fb in self._feature_browsers.items()},
            'region browser': self.region_browser.state(),
            'grid canvas': self.grid_canvas.state()
        }
        return data
