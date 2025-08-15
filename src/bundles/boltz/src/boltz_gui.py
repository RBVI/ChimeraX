# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
# Panel for searching AlphaFold or ESMFold databases or predicting structure
# from sequence.
#
from chimerax.core.tools import ToolInstance
class BoltzPredictionGUI(ToolInstance):
#    help = 'help:user/tools/boltz.html'
    help = 'help:boltz_help.html'

    def __init__(self, session, tool_name):

        self._auto_set_prediction_name = True
        self._boltz_runs = None		# List of BoltzRun instances if a prediction has been started
        self._installing_boltz = False
        self._affinity_value_map = {'each': 'each'}

        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self, close_destroys = False)
        tw.title = 'Boltz Structure Predicton'
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))

        '''
        heading = ('<html>'
                   f'Boltz structure prediction'
                   '<ul style="margin-top: 5;">'
                   '<li>Currently only handles protein, DNA, and RNA sequences.'
                   '</ul></html>')
        from Qt.QtWidgets import QLabel
        hl = QLabel(heading)
        layout.addWidget(hl)
        '''

        # Prediction name
        pn = self._create_prediction_name_entry(parent)
        layout.addWidget(pn.frame)
        
        # Make menu to choose molecules to predict
        sm = self._create_molecule_menu(parent)
        self._sequence_frame = sm
        layout.addWidget(sm)

        # Sequence entry field
        from Qt.QtWidgets import QTextEdit
        self._sequence_entry = se = QTextEdit(parent)
        se.setMaximumHeight(80)
        layout.addWidget(se)

        # Molecule assembly table
        self._molecules_table = None
        self._molecules_table_position = layout.count()

        # Progress bar
        self._progress_label = pl = self._create_progress_label(parent)
        layout.addWidget(pl)
        
        # Predict, Error Plot, Options, Help buttons
        bf = self._create_buttons(parent)
        layout.addWidget(bf)

        # Options panel
        options = self._create_options_gui(parent)
        layout.addWidget(options)

        layout.addStretch(1)    # Extra space at end

        self._update_entry_display()
        
        tw.manage(placement="side")

    # ---------------------------------------------------------------------------
    #
    @classmethod
    def get_singleton(cls, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, cls, 'Boltz', create=create)
        
    # ---------------------------------------------------------------------------
    #
    def _create_prediction_name_entry(self, parent):
        from chimerax.ui.widgets import EntriesRow

        pn = EntriesRow(parent, 'Prediction name', '')
        self._prediction_name = pname = pn.values[0]
        pname.pixel_width = 200
        pname._line_edit.textEdited.connect(self._prediction_name_edited)
        return pn

    # ---------------------------------------------------------------------------
    #
    def _prediction_name_edited(self, text):
        self._auto_set_prediction_name = (len(text) == 0)

    # ---------------------------------------------------------------------------
    #
    def _create_molecule_menu(self, parent):
        from Qt.QtWidgets import QFrame, QLabel, QPushButton, QMenu
        f = QFrame(parent)
        from chimerax.ui.widgets import horizontal_layout
        layout = horizontal_layout(f, margins = (2,0,0,0), spacing = 5)

        ab = QPushButton('Add', f)
        layout.addWidget(ab)
        ab.pressed.connect(self._add_molecule)

        ml = QLabel('molecule')
        layout.addWidget(ml)

        self._seq_button = mb = QPushButton(f)
        self._menu_data = {}	# Maps menu text to Chain, Structure or other associated menu entry data.
        layout.addWidget(mb)
        mb.pressed.connect(self._update_molecule_menu)
        self._mol_menu = m = QMenu(mb)
        mb.setMenu(m)
        m.triggered.connect(self._menu_selection_cb)
        entries = self._update_molecule_menu()
        if entries:
            mb.setText(entries[0][0])

        # UniProt entry field
        from Qt.QtWidgets import QLineEdit
        self._molecule_identifier_entry = ue = QLineEdit(f)
        ue.setMaximumWidth(200)
        layout.addWidget(ue)

        dr = QPushButton('Delete selected rows', f)
        layout.addWidget(dr)
        dr.pressed.connect(self._delete_selected_rows)

        cl = QPushButton('Clear', f)
        layout.addWidget(cl)
        cl.pressed.connect(self._clear_table)

        layout.addStretch(1)	# Extra space at end
        return f

    # ---------------------------------------------------------------------------
    #
    def _menu_selection_cb(self, action):
        text = action.text()
        self._seq_button.setText(text)
        self._update_entry_display()

    # ---------------------------------------------------------------------------
    #
    def _update_entry_display(self):
        text = self._seq_button.text()
        if text in ('protein sequence', 'rna sequence', 'dna sequence', 'ligand SMILES string',
                    'each ligand SMILES string'):
            show_seq, show_uniprot = True, False
        elif text in ('UniProt identifier', 'ligand CCD code'):
            show_seq, show_uniprot = False, True
        else:
            show_seq, show_uniprot = False, False
        self._sequence_entry.clear()
        self._molecule_identifier_entry.setText('')
        self._sequence_entry.setVisible(show_seq)
        self._molecule_identifier_entry.setVisible(show_uniprot)

    # ---------------------------------------------------------------------------
    #
    def _entry_strings(self, remove_whitespace = True, separator = ','):
        e = self._seq_button.text()
        if e in ('protein sequence', 'rna sequence', 'dna sequence', 'ligand SMILES string',
                 'each ligand SMILES string'):
            text = self._sequence_entry.toPlainText()
        elif e in ('UniProt identifier', 'ligand CCD code'):
            text = self._molecule_identifier_entry.text()
        else:
            text = ''
        strings = text.split(separator)
        if remove_whitespace:
            strings = [_remove_whitespace(string) for string in strings]
        strings = [string for string in strings if string.strip()]	# Remove empty strings
        return strings

    # ---------------------------------------------------------------------------
    #
    def _set_prediction_name(self):
        if not self._auto_set_prediction_name:
            return
        text = self._seq_button.text()
        if text in ('protein sequence', 'rna sequence', 'dna sequence',
                    'ligand CCD code', 'ligand SMILES string', 'each ligand SMILES string'):
            pname = ''
        elif text == 'UniProt identifier':
            uids = self._entry_strings()
            pname = uids[0] if len(uids) == 1 else ''
        else:
            s, c = self._menu_structure_or_chain()
            if s:
                pname = s.name
            elif c:
                pname = f'{c.structure.name}_{c.chain_id}'
            else:
                pname = ''
        if pname:
            pname.replace(' ', '_')
            self._prediction_name.value = pname

    # ---------------------------------------------------------------------------
    #
    def _menu_structure_or_chain(self):
        menu_text = self._seq_button.text()
        data = self._menu_data.get(menu_text)
        structure = chain = None
        from chimerax.atomic import Structure, Chain
        if isinstance(data, Structure):
            structure = data
        elif isinstance(data, Chain):
            chain = data
        return structure, chain

    # ---------------------------------------------------------------------------
    #
    def _update_molecule_menu(self):
        m = self._mol_menu
        m.clear()
        entries = self._menu_entries()
        for menu_text, data in entries:
            m.addAction(menu_text)
        self._menu_data = dict(entries)
        menu_text = self._seq_button.text()
        if menu_text not in self._menu_data:
            self._seq_button.setText(entries[0][0] if entries else '')
        return entries

    # ---------------------------------------------------------------------------
    #
    def _menu_entries(self):
        from chimerax.atomic import all_atomic_structures
        slist = all_atomic_structures(self.session)
        values = []
        for s in slist:
            values.extend(_specifiers_with_descriptions(s))
        values.extend([('protein sequence',None),
                       ('rna sequence',None),
                       ('dna sequence',None),
                       ('UniProt identifier',None),
                       ('ligand CCD code',None),
                       ('ligand SMILES string',None),
                       ('each ligand SMILES string',None)])
        return values

    # ---------------------------------------------------------------------------
    #
    def _add_molecule(self):
        self._update_molecule_menu()

        comps = self._new_components()
        if len(comps) == 0:
            return
        
        mt = self._molecules_table
        if mt is None:
            parent = self.tool_window.ui_area
            self._molecules_table = mt = MoleculesTable(parent, comps)
            layout = parent.layout()
            layout.insertWidget(self._molecules_table_position, mt)
            first_addition = True
        else:
            first_addition = (len(mt.data) == 0)
            mt.add_rows(comps)

        if first_addition:
            self._set_prediction_name()

        each_ligand = len([comp for comp in comps if comp.type == 'each ligand']) > 0
        if each_ligand and self._affinity_ligand_value() is None:
            self._affinity_ligand.value = 'each'
            
        self._report_number_of_tokens()
        
    # ---------------------------------------------------------------------------
    #
    def _new_components(self):
        comps = []
        e = self._seq_button.text()
        if e in ('protein sequence', 'rna sequence', 'dna sequence'):
            type = e.split()[0]
            for seq in self._entry_strings():
                seq_summary = f'{seq[:8]}...{seq[-8:]}' if len(seq) > 20 else seq
                desc = f'{type} sequence length {len(seq)}: {seq_summary}'
                comps.append(MolecularComponent(desc, type = type, sequence_string = seq))
        elif e == 'UniProt identifier':
            uniprot_errors = []
            from chimerax.atomic.args import UniProtSequenceArg, AnnotationError
            for uid in self._entry_strings():
                try:
                    seq = UniProtSequenceArg.parse(uid, self.session)[0]
                except AnnotationError as e:
                    uniprot_errors.append(str(e))
                else:
                    comps.append(MolecularComponent(f'UniProt sequence {uid}', type = 'protein',
                                                    uniprot_id = uid, sequence_string = seq.characters))
            if uniprot_errors:
                self.session.logger.error('\n'.join(uniprot_errors))
        elif e == 'ligand CCD code':
            from .predict import _ccd_atom_counts
            atom_counts = _ccd_atom_counts()
            unknown_ccds = []
            for ccd in self._entry_strings():
                if atom_counts is None or ccd in atom_counts:
                    comps.append(MolecularComponent(f'{ccd}', type = 'ligand', ccd_code = ccd))
                else:
                    unknown_ccds.append(ccd)
            if unknown_ccds:
                self.session.logger.error(f'Boltz does not have CCD code {", ".join(unknown_ccds)} in its CCD database ~/.boltz/ccd.pkl.  You could instead try to include that ligand using a SMILES string.')
        elif e == 'ligand SMILES string':
            for smiles in self._entry_strings():
                comps.append(MolecularComponent(f'ligand SMILES {smiles}', type = 'ligand', smiles_string = smiles))
        elif e == 'each ligand SMILES string':
            named_ligands = self._parse_each_ligand_smiles()  # list of (name, smiles)
            if named_ligands:
                comps.append(MolecularComponent(f'Each of {len(named_ligands)} ligands',
                                                type = 'each ligand', named_ligands = named_ligands))
        else:
            s, c = self._menu_structure_or_chain()
            if s:
                comps = [MolecularComponent(desc, type = polymer_type, chains = chains, count = len(chains),
                                            sequence_string = chains[0].characters)
                         for chains,polymer_type,desc in _unique_chain_descriptions(s.chains)]
                from .predict import _ccd_ligands_from_residues, _ccd_descriptions
                ccd_ligands, covalent_ligands = _ccd_ligands_from_residues(s.residues, exclude_ligands = ['HOH'])
                ccd_descrip = _ccd_descriptions(s)
                for ccd, count in ccd_ligands:
                    descrip = f'{ccd} - {ccd_descrip[ccd]}' if ccd in ccd_descrip else ccd
                    comps.append(MolecularComponent(descrip, count = count, type = 'ligand', ccd_code = ccd))
            elif c:
                comps = [MolecularComponent(desc, type = polymer_type, chains = chains, count = len(chains),
                                            sequence_string = chains[0].characters)
                         for chains,polymer_type,desc in _unique_chain_descriptions([c])]
        return comps

    # ---------------------------------------------------------------------------
    #
    def _parse_each_ligand_smiles(self):
        named_ligands = []
        rows = self._entry_strings(remove_whitespace = False, separator = '\n')
        for i, name_smiles in enumerate(rows):
            if name_smiles:
                fields = name_smiles.split(',')
                if len(fields) == 2:
                    named_ligands.append(tuple(f.strip() for f in fields))
                elif len(fields) == 1:
                    named_ligands.append((f'ligand{i+1}', fields[0].strip()))
                else:
                    msg = f'Each ligand SMILES text must have on each line one SMILES string or a ligand name and SMILES string separated by a comma, "{name_smiles}" is not valid.'
                    self.session.logger.error(msg)
                    return ()
        return named_ligands

    # ---------------------------------------------------------------------------
    #
    def _report_number_of_tokens(self):
        nres = natom = 0
        unknown_seqs = []
        unknown_ccds = []
        mt = self._molecules_table
        for comp in mt.data:
            if comp.type in ('protein', 'dna', 'rna'):
                if comp.sequence_string:
                    nres += len(comp.sequence_string) * comp.count
                else:
                    unknown_seqs.append(comp.uniprot_id)
            elif comp.type == 'ligand':
                if comp.ccd_code:
                    from .predict import _ccd_atom_count
                    na = _ccd_atom_count(comp.ccd_code)
                    if na is None:
                        unknown_ccds.append(comp.ccd_code)
                    else:
                        natom += na * comp.count
                elif comp.smiles_string:
                    from .predict import _smiles_atom_count
                    natom += _smiles_atom_count(comp.smiles_string) * comp.count

        lines = []
        if nres > 0:
            lines.append(f'{nres} polymer residues')
        if unknown_seqs:
            lines.append(f'Unknown sequence length {" ".join(unknown_seqs)}')
        if natom > 0:
            lines.append(f'{natom} ligand atoms')
        if unknown_ccds:
            lines.append(f'unknown CCD codes {" ".join(unknown_ccds)}')
        if len(unknown_seqs) == 0 and len(unknown_ccds) == 0 and nres > 0 and natom > 0:
            lines.append(f'{nres + natom} tokens')
        msg = ', '.join(lines)

        self._progress_label.setText(msg)
        
    # ---------------------------------------------------------------------------
    #
    def _delete_selected_rows(self):
        mt = self._molecules_table
        if mt:
            mt.delete_selected_rows()

        self._report_number_of_tokens()        
        
    # ---------------------------------------------------------------------------
    #
    def _clear_table(self):
        mt = self._molecules_table
        if mt:
            mt.clear()

    # ---------------------------------------------------------------------------
    #
    def _assembly_options(self):
        mt = self._molecules_table
        if mt is None or len(mt.data) == 0:
            return ''

        options = []
        for comp in mt.data:
            if comp.type in ('protein', 'dna', 'rna'):
                spec = comp.uniprot_id or comp.sequence_string
                for i in range(comp.count):
                    if comp.chains:
                        chain = comp.chains[i % len(comp.chains)]
                        if chain.chain is not None:  # Check if chain is deleted
                            spec = chain.atomspec
                    options.append(f'{comp.type} {spec}')

        ligand_ccds = []
        ligand_smiles = []
        for comp in mt.data:
            if comp.type == 'ligand':
                if comp.ccd_code:
                    ligand_ccds.append((comp.ccd_code, comp.count))
                elif comp.smiles_string:
                    ligand_smiles.append((comp.smiles_string, comp.count))
        if ligand_ccds:
            ccd_specs = _ligands_with_counts(ligand_ccds)
            options.append(f'ligandCcd {ccd_specs}')
        if ligand_smiles:
            smiles_specs = _ligands_with_counts(ligand_smiles)
            options.append(f'ligandSmiles {smiles_specs}')

        each_ligand = []
        for comp in mt.data:
            if comp.type == 'each ligand':
                each_ligand.extend([f'{name},{smiles}' for name, smiles in comp.named_ligands])
        if each_ligand:
            eligands = ','.join(each_ligand)
            options.append(f'forEachSmilesLigand "{eligands}"')
                            
        return options

    # ---------------------------------------------------------------------------
    #
    def _affinity_ligand_value(self):
        text = self._affinity_ligand.value
        if text == 'none':
            value = None
        elif text == 'last ligand':
            value = self._last_ligand()
        else:
            value = self._affinity_value_map.get(text, text)
        return value

    # ---------------------------------------------------------------------------
    #
    def _last_ligand(self):
        mt = self._molecules_table
        if mt is None or len(mt.data) == 0:
            return None
        ligand_spec = None
        for comp in mt.data:
            if comp.type == 'ligand':
                if comp.ccd_code:
                    ligand_spec = comp.ccd_code
                elif comp.smiles_string:
                    ligand_spec = comp.smiles_string
        return ligand_spec

    # ---------------------------------------------------------------------------
    #
    def _create_progress_label(self, parent):
        from Qt.QtWidgets import QLabel
        pl = QLabel(parent)
        return pl
    
    # ---------------------------------------------------------------------------
    #
    def _create_buttons(self, parent):
        buttons = [
            ('Predict', self._predict),
            ('Error plot', self._error_plot),
            ('Options', self._show_or_hide_options),
            ('Help', self._show_help),
        ]
        show_install = self._need_to_install_boltz()
        if show_install:
            buttons.insert(0, ('Install Boltz', self._install_boltz))
        from chimerax.ui.widgets import button_row
        f, buttons = button_row(parent, buttons, spacing = 5, button_list = True)
        self._button_row = f
        self._install_boltz_button = buttons[0] if show_install else None
        return f

    # ---------------------------------------------------------------------------
    #
    def _need_to_install_boltz(self):
        from .settings import _boltz_settings
        settings = _boltz_settings(self.session)
        boltz_dir = settings.boltz22_install_location
        from .install import find_executable
        boltz_exe = find_executable(boltz_dir, 'boltz')
        from os.path import isdir, isfile
        return not boltz_dir or not isdir(boltz_dir) or not isfile(boltz_exe)

    # ---------------------------------------------------------------------------
    #
    def _show_help(self):
        from chimerax.core.commands import run
        run(self.session, 'help %s' % self.help)
        
    # ---------------------------------------------------------------------------
    #
    def _predict(self):
        if self._installing_boltz:
            self.session.logger.error('Cannot make a prediction until Boltz installation finishes.')
            return
        if self._boltz_run and not self._boltz_run.finished:
            self.session.logger.error('Cannot make a new prediction until the current prediction finishes.')
            return
        options = []
        name = self._prediction_name.value
        if name:
            options.append(f'name {name}')
        dir = self._results_directory.value
        if dir != self.default_results_directory():
            options.append(f'resultsDirectory {dir}')
        if not self._use_msa_cache.value:
            options.append('useMsaCache false')
        if self._device.value != 'default':
            options.append(f'device {self._device.value}')
        lig = self._affinity_ligand_value()
        if lig:
            options.append(f'affinity {lig}')
        if self._use_steering_potentials.value:
            options.append('steering true')
        if self._use_cuda_bfloat16 and self._use_cuda_bfloat16.value:
            options.append('float16 true')
        if self._samples.value != 1:
            options.append(f'samples {self._samples.value}')
        from .settings import _boltz_settings
        settings = _boltz_settings(self.session)
        if self._install_directory.value != settings.boltz22_install_location:
            from chimerax.core.commands import quote_path_if_necessary
            options.append(f'installLocation {quote_path_if_necessary(self._install_directory.value)}')
        self._run_prediction(options = ' '.join(options))

    # ---------------------------------------------------------------------------
    #
    def _run_prediction(self, options = ''):
        assem_opt = self._assembly_options()
        if len(assem_opt) == 0:
            self.warn('No molecules specified for Boltz prediction')
            return
        assembly = ' '.join(assem_opt)
        
        cmd = f'boltz predict {assembly}'

        if options:
            cmd += f' {options}'

        from chimerax.core.commands import run
        br = run(self.session, cmd)
        if br is None:
            return  # Boltz not yet installed or other startup error.
        self._boltz_runs = br

        self._show_prediction_progress()

    # ---------------------------------------------------------------------------
    #
    @property
    def _boltz_run(self):
        br = self._boltz_runs
        if isinstance(br, list):
            while len(br) > 1 and br[0].finished:
                del br[0]
            br = br[0]
        return br

    # ---------------------------------------------------------------------------
    #
    def _show_stop_button(self, show):
        if show:
            br = self._button_row
            from Qt.QtWidgets import QPushButton
            self._stop_button = sb = QPushButton('Stop', br)
            br.layout().insertWidget(0, sb)
            sb.pressed.connect(self._stop_prediction)
        elif self._stop_button:
            sb = self._stop_button
            layout = self._button_row.layout()
            layout.removeWidget(sb)
            sb.deleteLater()
            self._stop_button = None

    # ---------------------------------------------------------------------------
    #
    def _stop_prediction(self):
        p = self._boltz_run
        if p:
            p.terminate()

    # ---------------------------------------------------------------------------
    #
    def _show_prediction_progress(self):
        from time import time
        self._prediction_start_time = t = time()
        self._next_progress_time = t + 1
        self._max_memory_use = None
        self.session.triggers.add_handler('new frame', self._report_progress)

        self._show_stop_button(True)
        
    # ---------------------------------------------------------------------------
    #
    def _report_progress(self, tname, tdata):
        if not self.tool_window.shown:
            return

        from time import time
        t = time()
        elapsed = t - self._prediction_start_time
        br = self._boltz_run
        if br and br.finished:
            status = 'completed in' if br.success else 'failed after'
            msg = f'Prediction {status} {"%.0f" % elapsed} seconds'
            if self._max_memory_use:
                msg += f', memory use {"%.1f" % self._max_memory_use} Gbytes'
            self._progress_label.setText(msg)
            self._show_stop_button(False)
            return 'delete handler'
        if t < self._next_progress_time:
            return
        self._next_progress_time = t + 1
        msg = f'Prediction running {"%.0f" % elapsed} seconds'
        if br.stage_info:
            msg += ': ' + br.stage_info
            if br.stage_info == 'sequence server busy... waiting':
                self._progress_label.setStyleSheet('background-color: lightyellow;')
                self._progress_label_colored = True
            elif getattr(self, '_progress_label_colored', False):
                self._progress_label.setStyleSheet('')
                self._progress_label_colored = False
                
        # Memory use values are not too useful on Mac ARM since GPU memory is not included
        # in RSS and too many libraries included in VMS.  Needs more investigation.
        '''
        import psutil
        try:
            p = psutil.Process(br._process.pid)
            # TODO: On Mac rss is 7.5 GB for 1hho dimer or 1hho tetramer.  It appears not to include GPU memory use.
            #       Wired memory use jumps to 20 GB for 1hho dimer, and sporadically down to 10 GB in Activity Monitor.
            #       I suspect the true GPU memory use is about 10 GB for 1hho dimer and that is the value that
            #       probably sets the maximum prediction size.  1hho dimer is about 380 tokens.  And the limit
            #       is about 1000 tokens on my 32 GB laptop.  Wired memory is about 2 GB before prediction.
            #       Peak wired memory use for 1hho tetramer 22 GB, was around 16 GB last 1/3 of prediction.
#            mem_gb = p.memory_info().rss / (1024 * 1024 * 1024)
#            mem_gb = p.memory_full_info().uss / (1024 * 1024 * 1024)  # PermissionError on Mac
#            mem_gb = p.memory_info().vms / (1024 * 1024 * 1024)	# ~400 GB on Mac
#            mem_gb = psutil.virtual_memory().used / (1024 * 1024 * 1024)  # 25 GB for 1hho dimer, too high by 2x.
            mem_gb = psutil.virtual_memory().wired / (1024 * 1024 * 1024)  # Tracks Activity Monitor closely.
#            mem_gb = 32 - psutil.virtual_memory().available / (1024 * 1024 * 1024) # Tracks Activity Monitor memory used 
            msg += f', memory use {"%.1f" % mem_gb} Gbytes'
            if self._max_memory_use is None or mem_gb > self._max_memory_use:
                self._max_memory_use = mem_gb
        except psutil.NoSuchProcess:
            pass
        '''
        self._progress_label.setText(msg)
        
    # ---------------------------------------------------------------------------
    #
    def _error_plot(self):
        self.show_error_plot()

    # ---------------------------------------------------------------------------
    #
    def show_error_plot(self):
        from chimerax.core.errors import UserError

        br = self._boltz_run
        if br is None:
            raise UserError('No Boltz prediction has been run.')

        if len(br._opened_predictions) == 0:
            raise UserError('Boltz predicted structure is not open.')

        structure = br._opened_predictions[0]
        if structure is None or structure.deleted:
            raise UserError('Boltz predicted structure is not open.')

        from chimerax.alphafold.pae import _matching_pae_file
        pae_path = _matching_pae_file(structure.filename)
        if pae_path is None:
            raise UserError(f'Could not find Boltz PAE file for "{structure.filename}".')
            
        from chimerax.alphafold.pae import AlphaFoldPAE, AlphaFoldPAEPlot
        pae = AlphaFoldPAE(pae_path, structure)
        p = AlphaFoldPAEPlot(self.session, 'Boltz Predicted Aligned Error', pae)
        
    # ---------------------------------------------------------------------------
    #
    def _create_options_gui(self, parent):
        from chimerax.ui.widgets import CollapsiblePanel
        self._options_panel = p = CollapsiblePanel(parent, title = None)
        f = p.content_area

        from .settings import _boltz_settings
        settings = _boltz_settings(self.session)

        from chimerax.ui.widgets import EntriesRow

        # Results directory
        rd = EntriesRow(f, 'Results directory', '',
                        ('Browse', self._choose_results_directory))
        self._results_directory = dir = rd.values[0]
        dir.pixel_width = 250
        dir.value = self.default_results_directory()

        # Number of predicted structures
        ns = EntriesRow(f, 'Number of predicted structures', 1)
        self._samples = sam = ns.values[0]
        sam.value = settings.samples

        # Affinity prediction
        la = EntriesRow(f, 'Predict ligand binding affinity for ', ('none', 'last ligand'))
        self._affinity_ligand = al = la.values[0]
        self._affinity_menu = am = al.widget.menu()
        am.aboutToShow.connect(self._fill_affinity_menu)

        # Steering potentials
        sp = EntriesRow(f, False, 'Use steering potentials.  May be more accurate, but slower.')
        self._use_steering_potentials = usp = sp.values[0]
        usp.value = settings.use_steering_potentials

        # Disabled this option since I only support it in Boltz 1.  Might add it to boltz 2 later.
        # Use 16-bit float with Nvidia CUDA, only shown if Nvidia gpu available
        from .install import have_nvidia_driver
        if have_nvidia_driver() and False:
            bf = EntriesRow(f, True, 'Predict larger structures with Nvidia 16-bit floating point')
            self._use_cuda_bfloat16 = cbf = bf.values[0]
            cbf.value = settings.use_cuda_bfloat16
        else:
            self._use_cuda_bfloat16 = None

        # Use MSA cache
        mc = EntriesRow(f, True, 'Use multiple sequence alignment cache')
        self._use_msa_cache = uc = mc.values[0]
        uc.value = settings.use_msa_cache

        # CPU or GPU device
        cd = EntriesRow(f, 'Compute device', ('default', 'cpu', 'gpu'))
        self._device = dev = cd.values[0]
        dev.value = settings.device
        
        # Boltz install location
        id = EntriesRow(f, 'Boltz install location', '',
                        ('Browse', self._choose_install_directory))
        self._install_directory = dir = id.values[0]
        dir.pixel_width = 350
        dir.value = settings.boltz22_install_location

        EntriesRow(f, ('Save default options', self._save_default_options))

        return p

        
    # ---------------------------------------------------------------------------
    #
    def _fill_affinity_menu(self):
        choices = ['none']
        self._affinity_value_map = {}	# Map abbreviated to full length smiles strings.
        mt = self._molecules_table
        if mt is not None:
            for comp in mt.data:
                if comp.type == 'ligand':
                    if comp.ccd_code:
                        choices.append(comp.ccd_code)
                    elif comp.smiles_string:
                        smiles = comp.smiles_string
                        if len(smiles) > 30:
                            smiles = smiles[:12] + '...' + smiles[-12:]
                            self._affinity_value_map[smiles] = comp.smiles_string
                        choices.append(smiles)
                elif comp.type == 'each ligand':
                    choices.append('each')
        choices.append('last ligand')
        menu = self._affinity_menu
        menu.clear()
        for text in choices:
            menu.addAction(text)
        
    # ---------------------------------------------------------------------------
    #
    def default_results_directory(self):
        from .settings import _boltz_settings
        settings = _boltz_settings(self.session)
        return settings.boltz_results_location

    # ---------------------------------------------------------------------------
    #
    def _show_or_hide_options(self):
        self._options_panel.toggle_panel_display()
        
    # ---------------------------------------------------------------------------
    #
    def _choose_results_directory(self):
        dir = _existing_directory(self._results_directory.value)
        if not dir:
            dir = _existing_directory(self.default_results_directory())
        parent = self.tool_window.ui_area
        from Qt.QtWidgets import QFileDialog
        path, ftype  = QFileDialog.getSaveFileName(parent,
                                                   caption = f'Boltz prediction results directory',
                                                   directory = dir,
                                                   options = QFileDialog.Option.ShowDirsOnly)
        if path:
            self._results_directory.value = path

    # ---------------------------------------------------------------------------
    #
    def _save_default_options(self, install_dir_only = False):
        from .settings import _boltz_settings
        settings = _boltz_settings(self.session)
        if not install_dir_only:
            settings.boltz_results_location = self._results_directory.value
            settings.samples = self._samples.value
            settings.device = self._device.value
            if self._use_cuda_bfloat16 is not None:
                settings.use_cuda_bfloat16 = self._use_cuda_bfloat16.value
            settings.use_steering_potentials = self._use_steering_potentials.value                
            settings.use_msa_cache = self._use_msa_cache.value
        settings.boltz22_install_location = self._install_directory.value
        settings.save()
        
    # ---------------------------------------------------------------------------
    #
    def _install_boltz(self):

        from os.path import expanduser
        boltz_dir = expanduser('~/boltz22')
        param_dir = expanduser('~/.boltz')
        message = ('Do you want to install Boltz?\n\n'
                   'This will take about 4 Gbytes of disk space and ten minutes or more depending on network speed.'
                   f' Boltz and its required packages will be installed in folder {boltz_dir} (1 GByte)'
                   '  and its model parameters (4 GBytes) and chemical component dictionary (1.8 Gbytes)'
                   f' will be installed in {param_dir}')
        from chimerax.ui.ask import ask
        answer = ask(self.session, message, title = 'Install Boltz', help_url = 'help:user/tools/boltz.html')
        if answer == 'yes':
            self._show_installing_boltz()
            from chimerax.core.commands import run, quote_path_if_necessary
            bdir = quote_path_if_necessary(boltz_dir)
            bi = run(self.session, f'boltz install {bdir}')
            if bi.success is not None:
                self._boltz_install_finished(bi.success)
            else:
                bi.finished_callback = self._boltz_install_finished
                self._installing_boltz = True
        
    # ---------------------------------------------------------------------------
    #
    def _show_installing_boltz(self):
        '''Replace "Install Boltz" button with text "Installing Boltz..."'''
        layout = self._button_row.layout()
        layout.removeWidget(self._install_boltz_button)
        self._install_boltz_button.setVisible(False)
        from Qt.QtWidgets import QLabel
        self._installing_label = QLabel('Installing Boltz...')
        layout.insertWidget(0, self._installing_label)

    # ---------------------------------------------------------------------------
    #
    def _boltz_install_finished(self, success):
        layout = self._button_row.layout()
        layout.removeWidget(self._installing_label)
        self._installing_label.deleteLater()
        self._installing_label = None
        if success:
            self._save_default_options(install_dir_only = True)
        else:
            layout.insertWidget(0, self._install_boltz_button)
            self._install_boltz_button.setVisible(True)

        self._installing_boltz = False

    # ---------------------------------------------------------------------------
    #
    def _choose_install_directory(self):
        dir = _existing_directory(self._install_directory.value)
        parent = self.tool_window.ui_area
        from Qt.QtWidgets import QFileDialog
        path  = QFileDialog.getExistingDirectory(parent,
                                                 caption = f'Boltz installation directory',
                                                 directory = dir,
                                                 options = QFileDialog.Option.ShowDirsOnly)
        if path:
            self._install_directory.value = path
            
    # ---------------------------------------------------------------------------
    #
    def warn(self, message):
        log = self.session.logger
        log.warning(message)
        log.status(message, color='red')

# -----------------------------------------------------------------------------
#
def _specifiers_with_descriptions(structure):
    values = []

    # Structure
    from chimerax.atomic import Residue
    polymers = [c for c in structure.chains if c.polymer_type != Residue.PT_NONE]
    if len(polymers) > 0:
        spec = structure.atomspec
        desc = ''
        if len(polymers) == 1:
            desc = _chain_description(polymers[0])
        menu_text = f'{spec} {desc}' if desc else spec
        values.append((menu_text, structure))

    # Chains
    if len(polymers) > 1:
        values.extend(_chain_descriptions(structure))

    return values

# -----------------------------------------------------------------------------
#
def _chain_description(chain):
    desc = chain.description
    if desc in ('', '.', None):
        return ''
    from chimerax.pdb import process_chem_name
    chain_desc = process_chem_name(desc)
    return chain_desc

# -----------------------------------------------------------------------------
#
def _chain_descriptions(structure):
    values = []
    from chimerax.atomic import Residue
    polymers = [c for c in structure.chains if c.polymer_type != Residue.PT_NONE]
    for c in sorted(polymers, key = lambda c: c.chain_id):
        spec = f'{structure.atomspec}/{c.chain_id}'
        desc = _chain_description(c)
        menu_text = f'{spec} {desc}' if desc else spec
        values.append((menu_text, c))
    return values

# -----------------------------------------------------------------------------
#
def _unique_chain_descriptions(chains):
    uchains = {}
    from chimerax.atomic import Residue
    for chain in chains:
        if chain.polymer_type == Residue.PT_AMINO:
            polymer_type = 'protein'
        elif chain.polymer_type == Residue.PT_NUCLEIC:
            # TODO: This is not reliable to distinguish RNA from DNA
            polymer_type = 'rna' if 'U' in chain.characters else 'dna'
        else:
            continue
        key = (polymer_type, chain.characters)
        if key in uchains:
            uchains[key].append(chain)
        else:
            uchains[key] = [chain]

    chain_descriptions = []
    for (polymer_type, seq), chains in uchains.items():
        from chimerax.atomic import concise_chain_spec
        cspec = concise_chain_spec(chains)
        desc = f'{polymer_type} {cspec}, length {len(seq)}'
        for c in chains:
            cdesc = _chain_description(c)
            if cdesc:
                desc = f'{cdesc}, {desc}'
                break
        chain_descriptions.append((chains, polymer_type, desc))

    return chain_descriptions

# -----------------------------------------------------------------------------
#
def _ligands_with_counts(ligand_specs):
    spec_counts = {}
    for spec, count in ligand_specs:
        if spec in spec_counts:
            spec_counts[spec] += count
        else:
            spec_counts[spec] = count
    lc = ','.join((spec if count == 1 else f'{spec}({count})')
                  for spec, count in spec_counts.items())
    return lc

# -----------------------------------------------------------------------------
#
def _existing_directory(directory):
    from os.path import expanduser, isdir, dirname
    dir = expanduser(directory)
    if dir == '' or isdir(dir):
        return directory
    return _existing_directory(dirname(dir))

# -----------------------------------------------------------------------------
#
def _remove_whitespace(string):
    from string import whitespace
    return string.translate(str.maketrans('', '', whitespace))

# -----------------------------------------------------------------------------
#
from chimerax.ui.widgets import ItemTable
class MoleculesTable(ItemTable):
    def __init__(self, parent, row_data = []):
        ItemTable.__init__(self, parent = parent, allow_user_sorting = False, auto_multiline_headers = False)
        self.setWordWrap(False)	# This allows ellipses to terminate long strings not limited to word boundaries.
        desc_col = self.add_column('Molecular component', 'short_description', justification = 'left')
        count_col = self.add_column('Count', 'count', format = '%d')
        self.data = row_data
        self.launch()

        # Keep the count column at fixed size.
        self.setColumnWidth(self.columns.index(count_col), 60)

        # Make the description column expand to maximum size.
        # Unfortunatly on my Mac Studio this expanded the width the tool panel.
#        from Qt.QtWidgets import QHeaderView
#        self.horizontalHeader().setSectionResizeMode(self.columns.index(desc_col), QHeaderView.Stretch)

    def add_rows(self, entries, scroll = True):
        uentries = self._remove_duplicates(self.data + entries)
        scroll = (scroll and len(uentries) > len(self.data))
        self.data = uentries
        if scroll:
            self.scrollToBottom()

    def _remove_duplicates(self, entries):
        uentries = {}
        for e in entries:
            key = e.unique_id()
            f = uentries.get(key)
            if f:
                f.count += e.count
                f.chains += e.chains
                uentries[key] = f.copy()  # ItemTable won't update display unless new entry instance found.
            else:
                uentries[key] = e
        return list(uentries.values())
            
    def delete_selected_rows(self):
        sel = set(self.selected)
        if sel:
            self.data = [d for d in self.data if d not in sel]

    def clear(self):
        self.data = []

# -----------------------------------------------------------------------------
#
class MolecularComponent:
    def __init__(self, description, type = None, count = 1,
                 chains = [], sequence_string = None, uniprot_id = None,
                 ccd_code = None, smiles_string = None, named_ligands = None):
        self.description = description
        self.type = type		# protein, dna, rna, ligand, each ligand
        self.count = count
        self.chains = chains		# chains of open models, use chain ids for prediction
        self.sequence_string = sequence_string
        self.uniprot_id = uniprot_id
        self.ccd_code = ccd_code
        self.smiles_string = smiles_string
        self.named_ligands = named_ligands	# Used for type "each ligand"

    def copy(self):
        return MolecularComponent(self.description, type = self.type, count = self.count, chains = self.chains,
                                  sequence_string = self.sequence_string, uniprot_id = self.uniprot_id,
                                  ccd_code = self.ccd_code, smiles_string = self.smiles_string)
    
    def unique_id(self):
        '''Used for combining rows for identical molecules.'''
        mspec = {
            'protein': ['sequence_string', 'uniprot_id'],
            'dna': ['sequence_string'],
            'rna': ['sequence_string'],
            'ligand': ['ccd_code', 'smiles_string'],
         }
        if self.type == 'each ligand':
            uid = id(self)	# Don't combine different each ligand specifiers
        else:
            uid = (self.type,) + tuple((attr, getattr(self, attr)) for attr in mspec[self.type])
        return uid

    @property
    def short_description(self, max_length=70):
        d = self.description
        if len(d) > max_length:
            d = d[:max_length] + '...'
        return d

# ------------------------------------------------------------------------------
#
from chimerax.core.tools import ToolInstance
class LigandPredictionsTable(ToolInstance):
    def __init__(self, session, predictions_directory, rows = None, smiles = None, align_to = None):
        self._predictions_directory = predictions_directory
        self._smiles = self._read_ligand_smiles() if smiles is None else smiles
        self._opened = [] if align_to is None else [align_to]
        ToolInstance.__init__(self, session, 'Boltz Ligand Predictions')

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        tw.fill_context_menu = self._fill_context_menu
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))

        # Heading
        from chimerax.ui.widgets import EntriesRow
        hl = EntriesRow(parent, 'Boltz predicted structures with ligands')
        layout.addWidget(hl.frame)

        # Table
        if rows is None:
            rows = self._table_rows()
        hl.labels[0].setText(f'Boltz predicted structures for {len(rows)} ligands')
        self._ligands_table = lt = BoltzLigandsTable(parent, rows)
        layout.addWidget(lt)

        # Buttons
        from chimerax.ui.widgets import button_row
        bf = button_row(parent,
                        [('Open', self._open_selected),],
                        spacing = 2)
        bf.setContentsMargins(0,5,0,0)
        layout.addWidget(bf)

        from .boltz_gui import boltz_panel
        bpanel = boltz_panel(self.session)
        placement = bpanel.tool_window if bpanel else 'right'
        tw.manage(placement = placement)

    def ligand_count(self):
        return len(self._ligands_table.data)

    def _table_rows(self):
        rows = []
        pdir = self._predictions_directory
        import os
        ligand_names = os.listdir(pdir)
        for ligand_name in ligand_names:
            from .predict import _read_json
            confidence = _read_json(pdir, ligand_name, f'confidence_{ligand_name}_model_0.json')
            if not confidence:
                continue  # Prediction failed probably due to chembl ligand standardization
            lig_iptm = confidence.get('ligand_iptm')
            affinity = _read_json(pdir, ligand_name, f'affinity_{ligand_name}.json')
            if affinity:
                log_affinity_uM = affinity.get('affinity_pred_value')
                affinity_uM = 10.0 ** log_affinity_uM
                prob = affinity.get('affinity_probability_binary')
            else:
                affinity_uM = prob = None
            smiles = self._smiles.get(ligand_name) if self._smiles else None
            row = LigandsTableRow(ligand_name, lig_iptm, affinity_uM, prob, smiles=smiles)
            rows.append(row)
        return rows

    def _read_ligand_smiles(self):
        from os.path import join, exists
        ligands_path = join(self._predictions_directory, '..', '..', 'ligands')
        if not exists(ligands_path):
            return None
        smiles = {}
        with open(ligands_path, 'r') as f:
            for line in f.readlines():
                fields = line.split(',')
                if len(fields) == 2 and fields[0] == 'smiles:':
                    smiles[fields[0].strip()] = fields[1].strip()
        return smiles
    
    def _open_selected(self):
        rows = self._ligands_table.selected		# LigandsTableRow instances
        paths = []
        missing_paths = []
        from os.path import join, exists
        for row in rows:
            name = row.ligand_name
            i = 0
            while True:
                path = join(self._predictions_directory, name, f'{name}_model_{i}.cif')
                if exists(path):
                    paths.append(path)
                    i += 1
                else:
                    if i == 0:
                        missing_paths.append(path)
                    break

        if missing_paths:
            self.session.logger.error(f'Could not find Boltz predictions {", ".join(missing_paths)}')

        if len(paths) == 0:
            return []
        
        # Open models
        from chimerax.core.commands import run, quote_path_if_necessary, concise_model_spec
        qpaths = [quote_path_if_necessary(path) for path in paths]
        cmd = 'open ' + ' '.join(qpaths)
        models = run(self.session, cmd)

        # Align models
        self._opened = [m for m in self._opened if not m.deleted]
        malign = models if self._opened else models[1:]
        self._opened.extend(models)
        if len(self._opened) > 1:
            model_spec = concise_model_spec(self.session, malign)
            first_model_spec = concise_model_spec(self.session, self._opened[:1])
            cmd = f'mm {model_spec} to {first_model_spec} logParameters false'
            run(self.session, cmd)

        return models

    def save_csv_file(self, path):
        with open(path, 'w') as f:
            self._ligands_table.write_values(f, separator = ',')

    def _fill_context_menu(self, menu, x, y):
        from Qt.QtGui import QAction
        act = QAction("Save CSV or TSV File...", parent=menu)
        act.triggered.connect(lambda *args, tab=self._ligands_table: tab.write_values())
        menu.addAction(act)

# -----------------------------------------------------------------------------
#
def open_boltz_ligands_file(session, path, align_to = None):
    with open(path, 'r') as f:
        line = f.readline()
        headings = line.strip().split(',')
        heading_attr = {
            'Ligand': 'ligand_name',
            'ipTM': ('iptm_score', float),
            'Affinity (uM)': ('binding_affinity', float),
            'Binding probability': ('binding_probability', float),
            'SMILES': 'smiles',
            }
        rows = []
        while True:
            line = f.readline()
            if not line:
                break
            fields = line.strip().split(',')
            if len(fields) != len(headings):
                from chimerax.core.errors import UserError
                raise UserError(f'Boltz ligands file has line with {len(fields)} values but there are {len(headings)} headings: {line}')
            attr_values = {}
            for value, h in zip(fields, headings):
                attr = heading_attr.get(h)
                if attr:
                    attr_name, attr_type = (attr, None) if isinstance(attr, str) else attr
                    if attr_type:
                        value = attr_type(value) if value else None
                    attr_values[attr_name] = value
            row = LigandsTableRow(**attr_values)
            rows.append(row)

    from os.path import join, dirname, basename
    predictions_directory = join(dirname(path), f'boltz_results_{basename(dirname(path))}', 'predictions')
    lpt = LigandPredictionsTable(session, predictions_directory, rows = rows, align_to = align_to)
    return lpt

# -----------------------------------------------------------------------------
#
from chimerax.ui.widgets import ItemTable
class BoltzLigandsTable(ItemTable):
    def __init__(self, parent, rows):
        ItemTable.__init__(self, parent = parent)
        self.add_column('Ligand', 'ligand_name')
        iptm = self.add_column('ipTM', 'iptm_score', format = '%.2f')
        self.add_column('Affinity (uM)', 'binding_affinity', format = '%.2g')
        self.add_column('Binding probability', 'binding_probability', format = '%.2g')
        smiles = self.add_column('SMILES', 'smiles')
        self.data = rows
        self.launch()
        self.sort_by(iptm, self.SORT_DESCENDING)
        smiles_column_index = self.columns.index(smiles)
        smiles_column_width = 250
        self.setColumnWidth(smiles_column_index, smiles_column_width)
#        self.setAutoScroll(False)  # Otherwise click on Description column scrolls horizontally
#        from Qt.QtWidgets import QSizePolicy
#        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)  # Don't resize whole panel width

# -----------------------------------------------------------------------------
#
class LigandsTableRow:
    def __init__(self, ligand_name, iptm_score, binding_affinity, binding_probability, smiles = None):
        self.ligand_name = ligand_name
        self.iptm_score = iptm_score
        self.binding_affinity = binding_affinity
        self.binding_probability = binding_probability
        self.smiles = smiles

# -----------------------------------------------------------------------------
#
def boltz_panel(session, create = False):
    return BoltzPredictionGUI.get_singleton(session, create=create)
  
# -----------------------------------------------------------------------------
#
def show_boltz_panel(session):
    return boltz_panel(session, create = True)
