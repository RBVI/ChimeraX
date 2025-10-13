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

def boltz_predict(session, sequences = [], ligands = None, exclude_ligands = 'HOH',
                  protein = [], dna = [], rna = [],
                  ligand_ccd = [], ligand_smiles = [], for_each_smiles_ligand = [],
                  name = None, results_directory = None,
                  device = None, kernels = None, precision = None, float16 = False,
                  samples = 1, recycles = 3, seed = None,
                  affinity = None, steering = False,
                  msa_only = False, use_msa_cache = True, msa_cache_dir = '~/Downloads/ChimeraX/BoltzMSA',
                  open = True, install_location = None, wait = None):

    if install_location is not None:
        from .settings import _boltz_settings
        settings = _boltz_settings(session)
        settings.boltz22_install_location = install_location
        settings.save()

    if not _is_boltz_available(session):
        return None

    if wait is None:
        wait = False if session.ui.is_gui else True

    polymer_components, modeled_chains, unmodeled_chains = _polymer_components(sequences, protein, dna, rna)
    align_to = modeled_chains[0].structure if modeled_chains else None
    used_chain_ids = set(sum((pc.chain_ids for pc in polymer_components), []))
    ligand_components, covalent_ligands = _ligand_components(ligands, exclude_ligands.split(','),
                                                             ligand_ccd, ligand_smiles, used_chain_ids)
    molecular_components = polymer_components + ligand_components

    predict_affinity = _affinity_component(affinity, ligand_components)
    _split_affinity_ligand(predict_affinity, molecular_components)
   
    # Warn about unmodeled compnents
    if unmodeled_chains:
        msg = f'Chains {", ".join(unmodeled_chains)} not modeled because not protein/DNA/RNA'
        session.logger.warning(msg)
    if covalent_ligands:
        session.logger.info(f'Predicting covalent ligands not yet supported: {covalent_ligands}')

    for_each_ligand = _for_each_ligand(for_each_smiles_ligand, used_chain_ids, predict_affinity, session.logger)
    if for_each_ligand is None:
        p = BoltzPrediction(name, molecular_components, predict_affinity = predict_affinity, align_to = align_to)
        predictions = [p]
    else:
        predictions = _each_ligand_predictions(for_each_ligand, molecular_components, predict_affinity, align_to)
        if predict_affinity:
            _warn_about_multicopy_ligands(predictions, session.logger)
            
    br = BoltzRun(session, predictions, name = name, run_directory = results_directory,
                  samples = samples, recycles = recycles, seed = seed, use_steering_potentials = steering,
                  device = device, use_kernels = kernels, precision = precision, cuda_bfloat16 = float16,
                  msa_only = msa_only, use_msa_cache = use_msa_cache, msa_cache_dir = msa_cache_dir,
                  open = open, wait = wait)

    msa_run = _msa_run(session, name, molecular_components, msa_cache_dir, wait) if for_each_ligand else None
    if msa_run is None:
        br.start()
    else:
        _start_multiple_runs([msa_run, br], wait=wait)

    return br if msa_run is None else [msa_run, br]

# ------------------------------------------------------------------------------
#
def _polymer_components(sequences, protein, dna, rna):

    # Choose chain ids for sequences.
    seqs = []
    chain_ids = set()
    modeled_chains = []
    unmodeled_chains = []
    from chimerax.atomic import Chain
    for seq_list, type in ((sequences, None), (protein, 'protein'), (dna, 'dna'), (rna, 'rna')):
        for seq in seq_list:
            seq_string = seq.characters
            is_chain = isinstance(seq, Chain)
            polymer_type = _chain_type(seq) if is_chain else None  # protein, dna, rna or None
            if type is None:
                if is_chain:
                    if polymer_type is None:
                        unmodeled_chains.append(seq)
                        continue
                else:
                    polymer_type = 'protein'
            elif polymer_type != type:
                continue
            chain_id = _next_chain_id(chain_ids, seq.chain_id) if is_chain else None
            seqs.append([polymer_type, chain_id, seq_string])
            if is_chain:
                modeled_chains.append(seq)

    # Assign unspecified chain ids after requested chain ids.
    for pcs in seqs:
        polymer_type, chain_id, seq_string = pcs
        if chain_id is None:
            pcs[1] = _next_chain_id(chain_ids)

    # Combine repeat sequences
    useqs = {}
    for polymer_type, chain_id, seq_string in seqs:
        if (polymer_type, seq_string) in useqs:
            useqs[(polymer_type, seq_string)].append(chain_id)
        else:
            useqs[(polymer_type, seq_string)] = [chain_id]

    # Create BoltzMolecules
    polymer_components = [BoltzMolecule(polymer_type, chain_ids, sequence_string = seq_string)
                          for (polymer_type, seq_string), chain_ids in useqs.items()]

    return polymer_components, modeled_chains, unmodeled_chains

# ------------------------------------------------------------------------------
#
def _chain_type(chain):
    from chimerax.atomic import Residue
    if chain.polymer_type == Residue.PT_AMINO:
        polymer_type = 'protein'
    elif chain.polymer_type == Residue.PT_NUCLEIC:
        # TODO: This is not reliable to distinguish RNA from DNA
        polymer_type = 'rna' if 'U' in chain.characters else 'dna'
    else:
        polymer_type = None
    return polymer_type

# ------------------------------------------------------------------------------
#
def _ligand_components(ligands, exclude_ligands, ligand_ccd, ligand_smiles, used_chain_ids):

    if ligands:
        ccd_ligands, covalent_ligands = _ccd_ligands_from_residues(ligands, exclude_ligands)
        ccd_counts = dict(ccd_ligands)
    else:
        covalent_ligands = ''
        ccd_counts = {}

    for ccd, count in ligand_ccd:
        if ccd in ccd_counts:
            ccd_counts[ccd] += count
        else:
            ccd_counts[ccd] = count

    ligand_components = []
    for ccd, count in ccd_counts.items():
        chain_ids = [_next_chain_id(used_chain_ids) for i in range(count)]
        ligand_components.append(BoltzMolecule('ligand', chain_ids, ccd_code = ccd))

    for smiles, count in ligand_smiles:
        chain_ids = [_next_chain_id(used_chain_ids) for i in range(count)]
        ligand_components.append(BoltzMolecule('ligand', chain_ids, smiles_string = smiles))
        
    return ligand_components, covalent_ligands

# ------------------------------------------------------------------------------
#
def _for_each_ligand(name_and_smiles, used_chain_ids, predict_affinity, log):
    n = len(name_and_smiles)
    if n == 0:
        return None

    if predict_affinity == 'each':
        # Remove multi-component ligands
        multi_comp = [name for name, smiles in name_and_smiles if '.' in smiles]
        if multi_comp:
            msg = f'Boltz 2 can only predict affinity of ligands having one covalently connected component. Will split {len(multi_comp)} ligands with multiple components ("." in SMILES string): {", ".join(multi_comp)}.'
            log.warning(msg)

    chain_ids = [_next_chain_id(used_chain_ids)]
    ligands = [BoltzMolecule('ligand', smiles_string = smiles, name = name.replace(' ', '_'), chain_ids = chain_ids)
               for name, smiles in name_and_smiles]
    return ligands
        
# ------------------------------------------------------------------------------
#
def _each_ligand_predictions(for_each_ligand, molecular_components, predict_affinity, align_to):
    predictions = []
    for ligand in for_each_ligand:
        components = molecular_components + [ligand]
        if predict_affinity == 'each':
            _split_affinity_ligand(ligand, components)
            affinity = ligand
        else:
            affinity = predict_affinity
        if affinity and _ligand_copies(affinity, components) >= 2:
            affinity = None	# Boltz 2.2 can't predict affinity for multicopy ligands
        p = BoltzPrediction(ligand.name, components, predict_affinity = affinity, align_to = align_to)
        p.ligand_smiles_string = ligand.smiles_string
        predictions.append(p)
    return predictions

# ------------------------------------------------------------------------------
#
def _ligand_copies(ligand, components):
    count = 0
    for c in components:
        if c.type == 'ligand' and c.smiles_string == ligand.smiles_string:
            count += 1
    return count

# ------------------------------------------------------------------------------
#
def _warn_about_multicopy_ligands(predictions, log):
    no_affinity = [p.name for p in predictions if p._predict_affinity is None]
    if no_affinity:
        log.warning(f'Boltz cannot predict affinity for ligands present in more than one copy.  Not predicting affinity for {", ".join(no_affinity)}')

# ------------------------------------------------------------------------------
#
def _affinity_component(affinity, ligand_components):
    if affinity == 'each':
        return 'each'		# Predicting multiple structures with different ligands
    predict_affinity = None
    if affinity:
        for lig_comp in ligand_components:
            if affinity in (lig_comp.ccd_code, lig_comp.smiles_string):
                ncopies = len(lig_comp.chain_ids)
                if ncopies > 1:
                    from chimerax.core.errors import UserError
                    raise UserError(f'Affinity ligand {affinity} has {ncopies} copies.  Boltz 2 can only predict affinity for single-copy ligands.')
                predict_affinity = lig_comp
                break
        if predict_affinity is None:
            from chimerax.core.errors import UserError
            raise UserError(f'Affinity ligand {affinity} is not a component of the predicted structure')
    return predict_affinity
        
# ------------------------------------------------------------------------------
#
def _split_affinity_ligand(affinity_ligand, molecular_components, allow_split = True):
    if (isinstance(affinity_ligand, BoltzMolecule) and
        affinity_ligand.smiles_string is not None and
        '.' in affinity_ligand.smiles_string):
        smiles = affinity_ligand.smiles_string
        if allow_split:
            smiles_pieces = smiles.split('.')
            smiles_pieces.sort(key = len, reverse = True)
            affinity_ligand.smiles_string = smiles_pieces[0]
            used_chain_ids = set(sum((mc.chain_ids for mc in molecular_components), []))
            extra_ligands = [BoltzMolecule('ligand', [_next_chain_id(used_chain_ids)], smiles_string = smiles_piece)
                             for smiles_piece in smiles_pieces[1:]]
            molecular_components.extend(extra_ligands)
            return True
        else:
            from chimerax.core.errors import UserError
            raise UserError(f'Affinity ligand {smiles} has multiple components that are not covalently linked ("." in SMILES string).  Boltz 2 cannot predict affinity for multi-component ligands.')
    return False

# ------------------------------------------------------------------------------
#
def _msa_run(session, name, molecular_components, msa_cache_dir, wait):
    proteins = [mc for mc in molecular_components if mc.type == 'protein']
    protein_seqs = [mc.sequence_string for mc in proteins]
    if protein_seqs:
        msa_cache_files = _find_msa_cache_files(protein_seqs, msa_cache_dir)
        if not msa_cache_files:
            msa_name = f'{name}_msa' if name else 'msa'
            prediction = BoltzPrediction(msa_name, proteins)
            br = BoltzRun(session, prediction, msa_only = True, msa_cache_dir = msa_cache_dir,
                          wait = wait)
            return br
    return None

# ------------------------------------------------------------------------------
#
def _start_multiple_runs(boltz_runs, wait = False):
    # Run MSA server calculation before multiple ligand structure predictions.
    if wait:
        for br in boltz_runs:
            br.start(wait = True)
            if not br.success:
                break
    else:
        def run_next(boltz_runs, previous_run = None):
            if boltz_runs and (previous_run is None or previous_run.success):
                boltz_runs[0].start(finished_callback =
                                    lambda: run_next(boltz_runs[1:], previous_run = boltz_runs[0]))
        run_next(boltz_runs)

# ------------------------------------------------------------------------------
#
class BoltzPrediction:
    '''Set of molecular components for predicting a single structure with Boltz.'''
    def __init__(self, name, molecular_components, predict_affinity = None, align_to = None):
        self.name = name
        self._molecular_components = molecular_components  # List of BoltzMolecule
        self._predict_affinity = predict_affinity          # BoltzMolecule for affinity prediction
        self._align_to = align_to	                   # AtomicStructure to align prediction to.

    def yaml_filename(self, default_name = 'input'):
        if self.name is None:
            self.name = default_name
        return self.name + '.yaml'

    def yaml_input(self, msa_cache_directory = None):
        yaml_lines = ['version: 1',
                      'sequences:']

        # Create yaml for polymers
        msa_cache_files = self._msa_cache_files(msa_cache_directory)
        for mc in self._molecular_components:
            if mc.type in ('protein', 'dna', 'rna'):
                polymer_entry = [f'  - {mc.type}:',
                                 f'      id: [{", ".join(mc.chain_ids)}]',
                                 f'      sequence: {mc.sequence_string}']
                if msa_cache_files and mc.type == 'protein':
                    msa_path = msa_cache_files[mc.sequence_string]
                    polymer_entry.append(f'      msa: {msa_path}')
                yaml_lines.extend(polymer_entry)

        # Create yaml for ligands
        for mc in self._molecular_components:
            if mc.type == 'ligand':
                ligand_entry = [ '  - ligand:',
                                f'      id: [{", ".join(mc.chain_ids)}]']
                if mc.ccd_code:
                    spec = f'      ccd: "{mc.ccd_code}"'
                elif mc.smiles_string:
                    smiles = mc.smiles_string.replace('\\', '\\\\')  # Escape backslashes
                    spec = f'      smiles: "{smiles}"'
                ligand_entry.append(spec)
                yaml_lines.extend(ligand_entry)

        # Predict affinity
        if self._predict_affinity:
            ligand_chain_id = self._predict_affinity.chain_ids[-1]
            affinity_lines = [
                'properties:',
                '  - affinity:', 
                f'      binder: {ligand_chain_id}']
            yaml_lines.extend(affinity_lines)

        # Create yaml string
        yaml = '\n'.join(yaml_lines)

        return yaml

    def _msa_cache_files(self, msa_cache_directory):
        if msa_cache_directory:
            protein_seqs = [mc.sequence_string for mc in self._molecular_components if mc.type == 'protein']
            msa_cache_files = _find_msa_cache_files(protein_seqs, msa_cache_directory)
        else:
            msa_cache_files = []

        from os.path import dirname
        self.cached_msa_dir = dirname(tuple(msa_cache_files.values())[0]) if msa_cache_files else None

        return msa_cache_files

    @property
    def using_cached_msa(self):
        return self.cached_msa_dir is not None
    
    def _add_to_msa_cache(self, msa_directory, cache_directory):
        if self.cached_msa_dir:
            return False
        protein_seqs = [mc.sequence_string for mc in self._molecular_components if mc.type == 'protein']
        if len(protein_seqs) == 0:
            return False
        from os.path import exists, join
        if not exists(msa_directory):
            return False
        if self.name is None:
            return False
        csv_paths = [join(msa_directory, f'{self.name}_{i}.csv') for i in range(len(protein_seqs))]
        for csv_path in csv_paths:
            if not exists(csv_path):
                return False
        _add_to_msa_cache(self.name, protein_seqs, csv_paths, cache_directory)

    def _assembly_description(self):
        mol_comps = self._molecular_components
        pcomps = [mc for mc in mol_comps if mc.type == 'protein']
        ncomps = [mc for mc in mol_comps if mc.type in ('dna', 'rna')]

        parts = []
        for comps, type in [(pcomps, 'protein'), (ncomps, 'nucleic acid sequence')]:
            if len(comps) == 1:
                nres = len(comps[0].sequence_string) * len(comps[0].chain_ids)
                parts.append(f'{type} with {nres} residues')
            elif len(comps) > 1:
                rlen = sum(len(comp.sequence_string) * len(comp.chain_ids) for comp in comps)
                parts.append(f'{len(comps)} {type}s with {rlen} residues')

        ligands = [(mc.ccd_code or mc.smiles_string, len(mc.chain_ids))
                   for mc in mol_comps if mc.type == 'ligand']
        lig_descrip = ', '.join([(lig_spec if count == 1 else f'{lig_spec} ({count})')
                                 for lig_spec, count in sorted(ligands)])
        if lig_descrip:
            nlig = sum(count for lig_spec, count in ligands)
            parts.append(f'{nlig} ligands {lig_descrip}')

        assem_descrip = ', '.join(parts)
        return assem_descrip
    
    def open_predictions(self, session, predictions_directory, num_samples = 1, align = True, color = True):
        models = []
        for n in range(num_samples):
            pmodels = self._open_prediction(session, predictions_directory, n=n, align=align, color=color)
            models.extend(pmodels)
        return models

    def _open_prediction(self, session, predictions_directory, n=0, align = True, color = True):
        # Find path to predicted model
        from os.path import join, exists
        mmcif_path = join(predictions_directory, self.name, f'{self.name}_model_{n}.cif')
        if not exists(mmcif_path):
            session.logger.warning('Prediction file not found: %s' % mmcif_path)
            return

        # Open predicted model
        from chimerax.core.commands import quote_path_if_necessary, run
        path_arg = quote_path_if_necessary(mmcif_path)
        models = run(session, f'open {path_arg} logInfo false')

        # Align prediction to input model
        if align and self._align_to and not self._align_to.deleted:
            aspec = self._align_to.atomspec
            for model in models:
                run(session, f'matchmaker {model.atomspec} to {aspec} logParameters false', log = False)

        # Color by confidence
        if color:
            for model in models:
                run(session, f'color bfactor {model.atomspec} palette alphafold log False', log = False)

        return models

    def _copy_predictions(self, predictions_directory, to_dir):
        '''
        Boltz puts the mmcif and confidence file predictions 4 directories deep.
        Copy them to the same directory as the input yaml file for ease of use.
        '''
        # Find path to predicted model
        from os.path import join
        pdir = join(predictions_directory, self.name)
        from os import listdir
        from shutil import copy
        for filename in listdir(pdir):
            if filename.endswith('.npz') and not filename.startswith('pae_'):
                continue	# Don't copy pde, plddt and pre_affinity files.
            copy(join(pdir, filename), to_dir)
        
# ------------------------------------------------------------------------------
#
class BoltzMolecule:
    def __init__(self, type, chain_ids, sequence_string = None, ccd_code = None, smiles_string = None, name = None):
        self.type = type	# protein, dna, rna, ligand
        self.chain_ids = chain_ids
        self.sequence_string = sequence_string
        self.ccd_code = ccd_code
        self.smiles_string = smiles_string
        self.name = name

# ------------------------------------------------------------------------------
#
class BoltzRun:
    def __init__(self, session, structures, name = None, run_directory = None,
                 samples = 1, recycles = 3, seed = None, use_steering_potentials = False,
                 device = 'default', use_kernels = None, precision = None, cuda_bfloat16 = False,
                 msa_only = False, use_msa_cache = True, msa_cache_dir = '~/Downloads/ChimeraX/BoltzMSA',
                 open = True, wait = False):

        self._session = session
        self._predictions = [structures] if isinstance(structures, BoltzPrediction) else structures
        self.name = name
        self._samples = samples		# Number of predicted structures
        self._recycles = recycles	# Number of boltz recycling steps
        self._device = device		# gpu, cpu or default, or None (uses settings value)
        self._use_kernels = use_kernels	# whether to use cuequivariance module for triangle attention
        self._precision = precision	# "32", "bf16-mixed", "16", "bf16-true"
        self._cuda_bfloat16 = cuda_bfloat16	# Save memory using 16-bit instead of 32-bit float
        self._use_steering_potentials = use_steering_potentials
        self._seed = seed		# Random seed for computation
        self._open = open		# Whether to open predictions when boltz finishes.

        from os.path import abspath, isabs
        run_dir = abspath(run_directory) if run_directory and not isabs(run_directory) else None
        self._run_directory = run_dir	# Location of input and results files
        self._input_path = None		# YAML file path or directory of yaml files
        self._running = False		# Subprocess running
        self._finished = False
        self._finished_callback = None
        self._stage = ''		# String included in status messages
        self._stage_detail = ''		# Extra info for status message, which ligand in batch predictions
        self._stage_times = {}		# Stage name to elapsed time for that stage
        self._stage_start_time = None	# Start of the current stage.
        self._user_terminated = False
        self.success = None
        self._process = None
        self._wait = wait
        self._start_time = None
        self._opened_predictions = []	# AtomicStructure instances opened when job completes

        # MSA cache parameters
        self._msa_only = msa_only
        self._use_msa_cache = use_msa_cache
        from os.path import expanduser
        self._msa_cache_dir = expanduser(msa_cache_dir)
        self._use_msa_server = True
        self.cached_msa_dir = None

    def start(self, finished_callback = None):
        self._finished_callback = finished_callback
        self._write_yaml_input_files()
        self._write_ligands_file()
        self._run_boltz()

    @property
    def _settings(self):
        from .settings import _boltz_settings
        settings = _boltz_settings(self._session)
        return settings

    def _write_yaml_input_files(self):
        # Create yaml before making directory so directory is not created if yaml creation fails.
        self._run_directory = dir = self._unique_run_directory()

        if self.name is None:
            from os.path import basename
            self.name = basename(dir)

        msa_cache_dir = self._msa_cache_dir if self._use_msa_cache else None
        yaml = [(p.yaml_filename(self.name), p.yaml_input(msa_cache_dir)) for p in self._predictions]

        for i, (filename, yaml_text) in enumerate(yaml):
            if filename is None:
                name = self.name if len(yaml) == 1 else f'{self.name}_{i+1}'
                self._predictions[i].name = name
                filename = name + '.yaml'
            from os.path import join
            yaml_path = join(dir, filename)
            with open(yaml_path, 'w') as f:
                f.write(yaml_text)
                
        self._input_path = yaml_path if len(yaml) == 1 else dir
        self._use_msa_server = (self._use_msa_server and self._need_msa_server)

    def _write_ligands_file(self, filename = 'ligands'):
        name_and_smiles = [(p.name, p.ligand_smiles_string) for p in self._predictions
                           if hasattr(p, 'ligand_smiles_string')]
        if name_and_smiles:
            text = '\n'.join(f'{name},{smiles}' for name, smiles in name_and_smiles)
            from os.path import join
            with open(join(self._run_directory, filename), 'w') as f:
                f.write(text)

    def _unique_run_directory(self):
        dir = self._run_directory
        if dir is None:
            dir = self._settings.boltz_results_location

        from os.path import expanduser
        dir = expanduser(dir)

        rdir = self._add_directory_suffix(dir)

        from os.path import exists
        if not exists(rdir):
            from os import makedirs
            makedirs(rdir)
        else:
            msg = f'Boltz prediction directory {rdir} already exists.  Files will be overwritten.'
            self._session.logger.warning(msg)

        return rdir

    def _add_directory_suffix(self, dir):
        if '[N]' not in dir and '[name]' not in dir:
            return dir
            
        if self.name:
            dir = dir.replace('[name]', self.name)  # Handle old boltz tool that used [N] for the name.
            dir = dir.replace('[N]', self.name)  # Handle old boltz tool that used [N] for the name.
            from os.path import exists
            if not exists(dir):
                return dir
        else:
            dir = dir.replace('[name]', '[N]')  # No name given so just use numeric suffix

        if '[N]' not in dir:
            dir += '_[N]'
            
        for i in range(1,1000000):
            path = dir.replace('[N]', str(i))
            from os.path import exists
            if not exists(path):
                return path

        return dir

    @property
    def device(self):
        if self._device is None:
            self._device = self._settings.device
        if self._device == 'default':
            device = boltz_default_device(self._session)
        else:
            device = self._device
        return device

    def _run_boltz(self):
        self._running = True
        self._set_stage('starting Boltz')

        self._log_prediction_info()

        boltz_venv = self._settings.boltz22_install_location
        from .install import find_executable
        boltz_exe = find_executable(boltz_venv, 'boltz')

        command = [boltz_exe, 'predict', self._input_path]
        
        if self._use_msa_server:
            command.append('--use_msa_server')

        if self._msa_only:
            command.append('--msa_only')

        if self._use_steering_potentials:
            command.append('--use_potentials')

        command.extend(['--accelerator', self.device])
        if self._cuda_bfloat16 and self.device == 'gpu':
            command.append('--use_cuda_bfloat16')

        use_kernels = self._use_kernels
        if self._use_kernels is None:
            from sys import platform
            use_kernels = (self.device == 'gpu' and platform == 'linux')
        if not use_kernels:
            command.append('--no_kernels')

        if self._precision is not None:
            command.extend(['--precision', self._precision])

        if self._samples != 1:
            command.extend(['--diffusion_samples', str(self._samples)])
        if self._recycles != 3:
            command.extend(['--recycling_steps', str(self._recycles)])
        if self._seed is not None:
            command.extend(['--seed', str(self._seed)])

        from sys import platform
        if platform == 'darwin':
            env = {}
            # On Mac PyTorch uses MPS (metal performance shaders) but not all functions are implemented
            # on the GPU (Feb 10, 2025) so PYTORCH_ENABLE_MPS_FALLBACK=1 allows these to run on the CPU.
            env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            # On Mac the huggingface.co URLs get SSL certificate errors unless we setup
            # certifi root certificates.
            import certifi
            env["SSL_CERT_FILE"] = certifi.where()
        else:
            env = None

        # Save command to a file
        from os.path import join
        command_file = join(self._run_directory, 'command')
        self._command = cmd = ' '.join(command)
        with open(command_file, 'w') as f:
            f.write(cmd)

        from subprocess import Popen, PIPE
        from .install import _no_subprocess_window
        # To continue to run even if ChimeraX exits use start_new_session=True
        p = Popen(command, cwd = self._run_directory,
                  stdout = PIPE, stderr = PIPE, env=env,
                  creationflags = _no_subprocess_window())
        self._process = p
        from time import time
        self._start_time = time()

        self._monitor_boltz_output()
        
        if self._wait:
            self._check_process_completion()
        else:
            self._session.triggers.add_handler('new frame', self._check_process_completion)

    @property
    def _need_msa_server(self):
        for p in self._predictions:
            if not p.using_cached_msa:
                return True
        return False
    
    def _log_prediction_info(self):
        pred = self._predictions
        mol_descrip = pred[0]._assembly_description() if len(pred) == 1 else f'{len(pred)} ligands'
        device = self.device
        log = self._session.logger
        log.info(f'Running Boltz prediction of {mol_descrip} on {device}')

        if self._use_msa_server:
            msa_method = 'Using multiple sequence alignment server https://api.colabfold.com'
        elif len(pred) == 1:
            msa_method = f'Using cached multiple sequence alignment {pred[0].cached_msa_dir}'
        else:
            msa_method = f'Using cached multiple sequence alignment'
        log.info(msa_method)

    def _monitor_boltz_output(self):
        p = self._process
        self._stdout = ReadOutputThread(p.stdout)
        self._stderr = ReadOutputThread(p.stderr)

    def _check_process_completion(self, *trigger_args):
        p = self._process
        if self._wait:
            while p.poll() is None:
                self._check_boltz_output()
                from time import sleep
                sleep(1)
        else:
            self._check_boltz_output()
            if p.poll() is None:
                return		    # Process still running

        self._set_stage('')	    # Job finished

        stdout = ''.join(self._stdout.all_lines())
        stderr = ''.join(self._stderr.all_lines())

        self._process_completed(stdout, stderr)

        return 'delete handler'

    def _check_boltz_output(self):

        new_lines = self._stderr.new_lines()
        if len(new_lines) == 0:
            return

        if len(self._predictions) > 1:
            # Report ligand being docked in status messages.
            detail_func = lambda text: text[25:].strip()
        else:
            detail_func = lambda text: ''
            
        stages = [('SUBMIT', 'submitting sequence search'),
                  ('RATELIMIT', 'sequence server busy... waiting'),
                  ('PENDING', 'sequence search submitted'),
                  ('RUNNING', 'sequence search running'),
                  ('COMPLETE', 'sequence search finished'),
                  ('Loading Boltz structure prediction weights', 'loading weights'),
                  ('Finished loading Boltz structure prediction weights', ''),
                  ('Starting structure inference', 'structure inference'),
                  ('Begin structure inference ', ('structure inference', detail_func)),
                  ('End structure inference ', 'structure inference'),
                  ('Finished structure inference', ''),
                  ('Loading affinity prediction weights', 'loading affinity weights'),
                  ('Finished loading affinity prediction weights', ''),
                  ('Starting affinity inference', 'affinity inference'),
                  ('Begin affinity inference ', ('affinity inference', detail_func)),
                  ('End affinity inference ', 'affinity inference'),
                  ('Finished affinity inference', ''),
                  ]
        found_stage = None
        for line in reversed(new_lines):
            msgs = []
            for text, stage in stages:
                if text in line:
                    pos = line.find(text)
                    if isinstance(stage, tuple):
                        stage, detail_function = stage
                        detail = detail_function(line[pos:])
                    else:
                        detail = ''
                    msgs.append((pos, stage, detail))
            if msgs:
                found_stage, detail = max(msgs)[1:]
                break

        if found_stage:
            self._set_stage(found_stage, detail=detail)
            
    def _set_stage(self, stage, detail = ''):
        self._stage_detail = detail
        if stage == self._stage:
            return
        from time import time
        t = time()
        if self._stage_start_time is not None:
            cur_stage = self._stage
            st = self._stage_times
            st[cur_stage] = st.get(cur_stage, 0) + t - self._stage_start_time
        self._stage = stage
        self._stage_start_time = t

    @property
    def stage_info(self):
        return self._stage if not self._stage_detail else (self._stage + ' ' + self._stage_detail)
    
    def _process_completed(self, stdout, stderr):
        
        dir = self._run_directory
        from os.path import join
        with open(join(dir, 'stdout'), 'w') as f:
            f.write(stdout)
        with open(join(dir, 'stderr'), 'w') as f:
            f.write(stderr)

        msg = self._prediction_failed_message(self._process.returncode, stdout, stderr)
        if msg:
            self._session.logger.error(msg)
            success = False
        else:
            if self._msa_only:
                self._report_runtime()
            elif len(self._predictions) > 1:
                self._report_multi_prediction_results()
            else:
                # One prediction
                for p in self._predictions:
                    p._copy_predictions(self._predictions_directory, self._run_directory)
                self._report_confidence()
                if self._predictions[0]._predict_affinity:
                    self._report_affinity()
                self._report_runtime()
                self._cite()
                if self._open:
                    models = []
                    for p in self._predictions:
                        models.extend(p.open_predictions(self._session, self._predictions_directory,
                                                         num_samples = self._samples))
                    self._opened_predictions = models
            success = True

        for p in self._predictions:
            p._add_to_msa_cache(self._msa_directory, self._msa_cache_dir)

        self.success = success
        self._running = False
        self._finished = True
        if self._finished_callback:
            self._finished_callback()

    def _report_runtime(self):
        from time import time
        total = time() - self._start_time
        parts = []
        st = self._stage_times
        sut = st.get('starting Boltz', 0)
        parts.append(f'start boltz {"%.0f" % sut} sec')
        wait_t = st.get('sequence server busy... waiting', 0)
        seq_t = (st.get('submitting sequence search', 0)
                 + wait_t
                 + st.get('sequence search submitted', 0)
                 + st.get('sequence search running', 0))
        sst = f'sequence search {"%.0f" % seq_t} sec'
        if wait_t > 0:
            sst += f' (waiting {"%.0f" % wait_t} sec, running {"%.0f" % (seq_t-wait_t)} sec)'
        parts.append(sst)
        lwt = st.get('loading weights', 0)
        parts.append(f'load weights {"%.0f" % lwt} sec')
        sit = st.get('structure inference', 0)
        parts.append(f'structure inference {"%.0f" % sit} sec')
        if [p for p in self._predictions if p._predict_affinity]:
            lawt = st.get('loading affinity weights', 0)
            parts.append(f'load affinity weights {"%.0f" % lawt} sec')
            ait = st.get('affinity inference', 0)
            parts.append(f'affinity inference {"%.0f" % ait} sec')

        timings = ', '.join(parts)
        msg = f'Boltz prediction completed in {"%.0f" % total} seconds ({timings})'
        self._session.logger.info(msg)

        if self._use_msa_server and wait_t >= 60:
            msg = f'The sequence alignment server api.colabfold.com was busy and Boltz waited {"%.0f" % wait_t} seconds to start the alignment computation for this prediction.'
            self._session.logger.warning(msg)
            
    def _prediction_failed_message(self, exit_code, stdout, stderr):

        msg = None
        if self._prediction_ran_out_of_memory(stdout):
            msg = ('The Boltz prediction ran out of memory.  The memory use depends on the'
                   ' number of protein and nucleic acid residues plus the number of ligand'
                   ' atoms.  You can reduce the size of your molecular assembly to stay'
                   ' within the memory limits.')
        elif exit_code != 0 or (not self._msa_only and not self._predicted_model_exists()):
            if 'No supported gpu backend found' in stderr:
                msg = ('Attempted to run Boltz on the GPU but no supported GPU device could be found.'
                       ' To avoid this error specify the compute device as "cpu" in the ChimeraX Boltz'
                       ' options panel, or using the ChimeraX boltz command "device cpu" option.'
                       ' Boltz supports Nvidia GPUs with CUDA and Mac M series GPUs.  On Windows the Boltz'
                       ' installation installs torch with no GPU support, so using Nvidia GPUs on'
                       ' Windows requires reinstalling gpu-enabled torch with Boltz which we plan to'
                       ' support in the future.')
            elif 'No such option: --use_cuda_bfloat16' in stderr:
                msg = ('The installed Boltz does not have the --use_cuda_bfloat16 option.'
                       ' You need to install a newer Boltz to get this option.'
                       ' Use the ChimeraX command "boltz install ~/boltz_new" to install it.')
            elif 'No such option: --no_potentials' in stderr:
                msg = ('The installed Boltz does not have the --no_potentials option.'
                       ' You need to install a newer Boltz to get this option.'
                       ' Use the ChimeraX command "boltz install ~/boltz_new" to install it.')
            elif 'ValueError: Molecule is excluded' in stderr:
                msg = ('Affinity calculation of a ligand given as a SMILES string'
                       ' containing certain elements (e.g. metals) cannot be handled'
                       ' because the ChEMBL structure pipeline library used by Boltz'
                       ' cannot standardize such SMILES strings.')
            elif 'ValueError: CCD component ' in stderr:
                i_ccd_start = stderr.find('ValueError: CCD component ') + 26
                i_ccd_end = i_ccd_start + stderr[i_ccd_start:].find(' ')
                ccd_code = stderr[i_ccd_start:i_ccd_end]
                msg = ('Your Boltz installation does not have a molecular structure for'
                       f' PDB chemical component dictionary code {ccd_code} either because'
                       ' that code is new or is mistyped.  You can try specifying that ligand'
                       f' using a SMILES string from https://www.rcsb.org/ligand/{ccd_code}'
                       ' instead of using its CCD code.')
            elif 'load_from_checkpoint' in stderr and 'PytorchStreamReader failed reading zip archive' in stderr:
                msg = ('Boltz failed reading neural network weights from directory ~/.boltz.'
                       ' This can happen if you quit ChimeraX while installing Boltz before the'
                       ' installation finished. To fix it delete the ~/.boltz directory.'
                       ' The next time a Boltz prediction is run it will download the neural'
                       ' network weights and chemical components (5 Gbytes). That may take'
                       ' minutes to hours depending on your internet connection speed.')
            else:
                if self._user_terminated:
                    msg = 'Prediction terminated by user'
                    self._user_terminated = False
                else:
                    msg = '\n'.join([
                        f'Running boltz prediction failed with exit code {exit_code}:',
                        'command:',self._command,
                        'stdout:', stdout,
                        'stderr:', stderr,
                        ])
        return msg

    def _predicted_model_exists(self):
        from os.path import join, exists
        for p in self._predictions:
            # Not all ligand predictions succeed due to chembl standarization errors.
            mmcif_path = join(self._predictions_directory, p.name, f'{p.name}_model_0.cif')
            if exists(mmcif_path):
                return True
        return False
        
    @property
    def finished(self):
        return self._finished

    def terminate(self):
        if self._running:
            self._process.kill()
            self._user_terminated = True

    def _prediction_ran_out_of_memory(self, stdout):
        from os.path import join, exists
        pdir = join(self._predictions_directory, self.name)
        return not exists(pdir) and 'ran out of memory' in stdout

    @property
    def _results_directory(self):
        from pathlib import Path
        name = Path(self._input_path).stem
        from os.path import join
        return join(self._run_directory, f'boltz_results_{name}')

    @property
    def _predictions_directory(self):
        from os.path import join
        return join(self._results_directory, 'predictions')

    @property
    def _msa_directory(self):
        from os.path import join
        return join(self._results_directory, 'msa')

    def _report_confidence(self):
        from os.path import join, exists
        lines = []
        for sample in range(self._samples):
            results = _read_json(self._predictions_directory, self.name, f'confidence_{self.name}_model_{sample}.json')
            if not results:
                continue
            conf = results.get('confidence_score', -1)
            ptm = results.get('ptm', -1)
            iptm = results.get('iptm', -1)
            lig_iptm = results.get('ligand_iptm', -1)
            prot_iptm = results.get('protein_iptm', -1)
            plddt = results.get('complex_plddt', -1)

            if iptm != ptm:
                iptm_text = f'ipTM {"%.2f" % iptm}'
                if lig_iptm > 0 and prot_iptm > 0:
                    iptm_text += f' (ligand {"%.2f" % lig_iptm}, protein {"%.2f" % prot_iptm})'
            parts = [f'Confidence score {"%.2f" % conf}',
                     f'pTM {"%.2f" % ptm}',
                     iptm_text,
                     f'pLDDT {"%.2f" % plddt}']
            lines.append(', '.join(parts))
        if lines:
            self._session.logger.info('<br>'.join(lines), is_html = True)

    def _report_affinity(self):
        for p in self._predictions:
            ligand_mol = p._predict_affinity
            if ligand_mol:
                results = _read_json(self._predictions_directory, p.name, f'affinity_{p.name}.json')
                if results:
                    log_affinity_uM = results['affinity_pred_value']
                    affinity_uM = 10.0 ** log_affinity_uM
                    prob = results['affinity_probability_binary']
                    lig_spec = (ligand_mol.ccd_code or ligand_mol.smiles_string)
                    msg = f'Ligand {lig_spec} predicted binding affinity {"%.2g" % affinity_uM} uM, predicted binding probability {"%.2g" % prob}'
                    self._session.logger.info(msg)

    def _report_multi_prediction_results(self):
        n = len(self._predictions)
        msg = f'Predicted structures for {n} ligands'
        self._session.logger.info(msg)
        self._report_runtime()
        # Create table of results
        smiles = {p.name:p.ligand_smiles_string for p in self._predictions}
        align_to = self._predictions[0]._align_to if self._predictions else None
        from . import boltz_gui
        t = boltz_gui.LigandPredictionsTable(self._session, self._predictions_directory,
                                             smiles = smiles, align_to = align_to)
        self._save_results_as_csv_file(t)

    def _save_results_as_csv_file(self, table):
        from os.path import join
        csv_path = join(self._run_directory, f'{self.name}.bzlig')
        table.save_csv_file(csv_path)
        from chimerax.core.filehistory import remember_file
        remember_file(self._session, csv_path, 'bzlig', models=[], file_saved=True)

    def _cite(self):
        session = self._session
        if not hasattr(session, '_cite_boltz'):
            msg = 'Please cite <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC11601547/">Boltz-1 Democratizing Biomolecular Interaction Modeling. BioRxiv https://doi.org/10.1101/2024.11.19.624167</a> if you use these predictions.'
            session.logger.info(msg, is_html = True)
            session._cite_boltz = True  # Only log this message once per session.

# ------------------------------------------------------------------------------
#
def _read_json(*path_components):
    from os.path import join, exists
    path = join(*path_components)
    if not exists(path):
        return None
    import json
    with open(path, 'r') as f:
        results = json.load(f)
    return results

# ------------------------------------------------------------------------------
#
def _next_chain_id(used_already, chain_id = None):
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    suffixes = ['', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    if chain_id is None:
        for suffix in suffixes:
            for cid in letters:
                cid += suffix
                if cid not in used_already:
                    used_already.add(cid)
                    return cid
    else:
        for suffix in suffixes:
            cid = chain_id + suffix
            if cid not in used_already:
                used_already.add(cid)
                return cid

    raise RuntimeError(f'Could not assign unique chain id for {chain_id}')

# ------------------------------------------------------------------------------
#
def _chain_names(chains):
    sc = {}
    for chain in chains:
        s = chain.structure
        if s not in sc:
            sc[s] = []
        sc[s].append(chain.chain_id)
    return ''.join(str(s) + '/' + ','.join(schains) for s, schains in sc.items())

# ------------------------------------------------------------------------------
#
def boltz_ligand_table(session, run_directory, align_to = None):
    '''Show a table of Boltz ligand binding prediction results.'''
    from os.path import join, basename, exists
    predictions_directory = join(run_directory, f'boltz_results_{basename(run_directory)}', 'predictions')
    if not exists(predictions_directory):
        from chimerax.core.errors import UserError
        raise UserError(f'Expected to find Boltz predictions in results directory {predictions_directory} which does not exist.')
    from . import boltz_gui
    boltz_gui.LigandPredictionsTable(session, predictions_directory, align_to = align_to)

# ------------------------------------------------------------------------------
#
def _is_boltz_available(session):
    '''Check if Boltz is locally installed with paths properly setup.'''
    from .settings import _boltz_settings
    settings = _boltz_settings(session)
    install_location = settings.boltz22_install_location
    from os.path import isdir
    if not isdir(install_location):
        msg = 'You need to install Boltz by pressing the "Install Boltz" button on the ChimeraX Boltz user interface, or using the ChimeraX command "boltz install".  If you already have Boltz installed, you can set the Boltz installation location in the user interface under Options, or use the installLocation option of the ChimeraX boltz command "boltz predict ... installLocation /path/to/boltz"'
        session.logger.error(msg)
        return False

    if not _check_venv_valid(session, install_location):
        return False

    return True

# ------------------------------------------------------------------------------
#
def _check_venv_valid(session, install_location):
    '''
    Warn if the ChimeraX python that was used for boltz venv does not exist.
    '''
    from os.path import join, exists
    venv_config_path = join(install_location, 'pyvenv.cfg')
    if not exists(venv_config_path):
        # Maybe the user installed Boltz themselves.  Don't complain in that case.
        return True

    with open(venv_config_path, 'r') as f:
        # Prepend section name since configparser requires a section.
        config = '[params]\n' + f.read()
    import configparser
    p = configparser.ConfigParser()
    p.read_string(config)

    home = p.get('params', 'home', fallback = None)  # Python bin directory
    if home and not exists(home):
        from chimerax.core.commands import quote_path_if_necessary
        install_loc = quote_path_if_necessary(install_location)
        msg = f'The ChimeraX version you used to install Boltz no longer exists and Boltz uses the Python from that ChimeraX.  You need to reinstall Boltz with your current ChimeraX.  First remove the directory containing the old Boltz installation\n\n{install_location}\n\nThen restart ChimeraX and press the "Install Boltz" button on the ChimeraX Boltz panel or use the ChimeraX command\n\nboltz install {install_loc}'
        session.logger.error(msg)
        return False

    return True

# ------------------------------------------------------------------------------
#
def boltz_default_device(session):
    from sys import platform
    if platform in ('win32', 'linux'):
        from .install import have_nvidia_driver
        device = 'gpu' if have_nvidia_driver() and _torch_has_cuda(session) else 'cpu'
        # TODO: On Linux run nvidia-smi to see if GPU memory is sufficient to run Boltz.
    elif platform == 'darwin':
        from platform import machine
        device = 'gpu' if machine() == 'arm64' else 'cpu'
        # PyTorch 2.6 does not support Intel Mac GPU.
        #     https://discuss.pytorch.org/t/pytorch-support-for-intel-gpus-on-mac/151996
    else:
        device = 'cpu'
    return device

# ------------------------------------------------------------------------------
#
def _torch_has_cuda(session):
    from sys import platform
    if platform == 'darwin':
        return False
    if platform == 'win32':
        lib_path = 'Lib/site-packages/torch/lib/torch_cuda.dll'
    elif platform == 'linux':
        from sys import version_info as v
        lib_path = f'lib/python{v.major}.{v.minor}/site-packages/torch/lib/libtorch_cuda.so'
    from .settings import _boltz_settings
    settings = _boltz_settings(session)
    boltz_install = settings.boltz22_install_location
    from os.path import join, exists
    torch_cuda_lib = join(boltz_install, lib_path)
    return exists(torch_cuda_lib)

# ------------------------------------------------------------------------------
#
def _find_msa_cache_files(protein_seqs, msa_cache_dir):
    msa_cache_files = []
    from os.path import exists, join, splitext, expanduser
    msa_cache_dir = expanduser(msa_cache_dir)
    index_path = join(msa_cache_dir, 'index')
    if not exists(msa_cache_dir) or not exists(index_path):
        return msa_cache_files
    with open(index_path, 'r') as f:
        index_lines = f.readlines()
    for line in index_lines:
        fields = line.strip().split(',')
        msa_dir, msa_seqs = fields[0], fields[1:]
        if msa_seqs == protein_seqs:
            csv_files = _csv_files(join(msa_cache_dir, msa_dir))
            # Sort by integer file name suffix.
            csv_files.sort(key = lambda fname: int(splitext(fname)[0].split('_')[-1]))
            csv_paths = [join(msa_cache_dir, msa_dir, csv_file) for csv_file in csv_files]
            msa_cache_files = {seq:csv_path for seq, csv_path in zip(protein_seqs, csv_paths)}
            break
    return msa_cache_files

# ------------------------------------------------------------------------------
#
def _csv_files(dir):
    from os import listdir
    return [csv_file for csv_file in listdir(dir) if csv_file.endswith('.csv')]

# ------------------------------------------------------------------------------
#
def _add_to_msa_cache(dir_name, protein_seqs, csv_paths, msa_cache_dir):
    from os.path import exists, join, basename
    if not exists(msa_cache_dir):
        from os import makedirs
        makedirs(msa_cache_dir)

    new_cache_dir = join(msa_cache_dir, dir_name)
    if exists(new_cache_dir):
        new_cache_dir = _unique_cache_dir(new_cache_dir)
        dir_name = basename(new_cache_dir)

    from os import mkdir
    mkdir(new_cache_dir)
    from shutil import copy2
    for csv_path in csv_paths:
        copy2(csv_path, new_cache_dir)

    seqs_string = ','.join(protein_seqs)
    entry_line = f'{dir_name},{seqs_string}'
    index_path = join(msa_cache_dir, 'index')
    with open(index_path, 'a') as f:
        f.write('\n' + entry_line)

    return True

# ------------------------------------------------------------------------------
#
def _unique_cache_dir(dir):
    '''Add or increment numberic suffix separated by an underscore character.'''
    from os.path import exists
    if not exists(dir):
        return dir
    n = 0
    parts = dir.split('_')
    if len(parts) == 1:
        base = dir
    else:
        try:
            n = int(parts[-1])
        except:
            base = dir
        else:
            base = '_'.join(parts[:-1])
    while exists(f'{base}_{n}'):
        n += 1
    return f'{base}_{n}'

# ------------------------------------------------------------------------------
#
def _ccd_ligands_from_residues(residues, exclude_ligands = []):
    from chimerax.atomic import Residue
    npres = residues[residues.polymer_types == Residue.PT_NONE]
    cres, ncres = _covalently_linked_residues(npres)
    lig_counts = {}
    for r in ncres:
        rtype = r.name
        if rtype not in exclude_ligands:
            if rtype in lig_counts:
                lig_counts[rtype] += 1
            else:
                lig_counts[rtype] = 1
    ccd_ligands = [(rtype, count) for rtype, count in lig_counts.items()]

    from chimerax.atomic import concise_residue_spec
    if len(cres) > 0:
        session = cres[0].structure.session
        covalent_ligands = concise_residue_spec(session, cres)
    else:
        covalent_ligands =  ''

    return ccd_ligands, covalent_ligands

# ------------------------------------------------------------------------------
#
def _ccd_descriptions(structure):
    ccd_descrip = {}
    from chimerax.mmcif import get_mmcif_tables_from_metadata
    chem_comp = get_mmcif_tables_from_metadata(structure, ["chem_comp"])[0]
    if chem_comp:
        rows = chem_comp.fields(['id', 'name', 'pdbx_synonyms'], allow_missing_fields=True)
        for row in rows:
            if row[1]:
                from chimerax.pdb import process_chem_name
                ccd_descrip[row[0]] = process_chem_name(row[1])
    return ccd_descrip

# ------------------------------------------------------------------------------
#
def _covalently_linked_residues(residues):
    cres, ncres = [], []
    for r in residues:
        if len(r.neighbors) == 0:
            ncres.append(r)
        else:
            cres.append(r)
    from chimerax.atomic import Residues
    return Residues(cres), Residues(ncres)

# ------------------------------------------------------------------------------
#
def _ccd_atom_count(ccd_code):
    counts = _ccd_atom_counts()
    return counts.get(ccd_code) if counts else None

# ------------------------------------------------------------------------------
#
_ccd_atom_counts_table = None
def _ccd_atom_counts():
    global _ccd_atom_counts_table
    if _ccd_atom_counts_table is None:
        from os.path import expanduser, exists
        counts_path = expanduser('~/.boltz/ccd_atom_counts_boltz2.npz')
        if exists(counts_path):
            import numpy
            with numpy.load(counts_path) as counts:
                _ccd_atom_counts_table = dict(zip(counts['ccds'],counts['counts']))
    return _ccd_atom_counts_table

# ------------------------------------------------------------------------------
#
def _smiles_atom_count(smiles_string):
    atom_count = 0
    from chimerax.atomic import Element
    element_names = Element.names	# A set.
    n = len(smiles_string)
    i = 0
    while i < n:
        ename2 = smiles_string[i:i+2]
        if i < n-1 and ename2 in element_names:
            atom_count += 1
            if ename2 == 'Sc' and (i == 0 or smiles_string[i-1:i] != '['):
                i += 1  # This is sulfur and carbon, not scandium
            else:
                i += 2
        else:
            ename = smiles_string[i:i+1].upper()
            if ename in element_names and ename != 'H':
                atom_count += 1;
                i += 1
            elif ename == '@' and smiles_string[i+1:i+3] in ('TH', 'SP', 'AL', 'TB', 'OH'):
                i += 3  # Skip chiral center codes.
            else:
                i += 1
    return atom_count

# ------------------------------------------------------------------------------
#
def _test_smiles_atom_count():
    from os.path import expanduser
    ccd_path = expanduser('~/.boltz/ccd.pkl')
    import pickle
    with open(ccd_path, 'rb') as f:
        ccd_mols = pickle.load(f)
    from rdkit.Chem import MolToSmiles
    from rdkit.Chem.rdMolDescriptors import CalcNumHeavyAtoms
    for ccd, mol in ccd_mols.items():
        smiles = MolToSmiles(mol)
        sa_count = _smiles_atom_count(smiles)
        ccd_count = CalcNumHeavyAtoms(mol)
        if sa_count != ccd_count:
            print(f'Smiles {smiles} atom count {sa_count} differs from ccd atom count {ccd_count}.')

# ------------------------------------------------------------------------------
#
class ReadOutputThread:
    def __init__(self, stream):
        self._stream = stream

        import locale
        self._text_encoding = locale.getpreferredencoding()
        self._all_lines = []

        from queue import Queue
        self._queue = Queue()
        from threading import Thread
        # Set daemon true so that ChimeraX exit is not blocked by the thread still running.
        self._thread = t = Thread(target = self._queue_output_in_thread, daemon = True)
        t.start()

    def _queue_output_in_thread(self):
        while True:
            line = self._stream.readline() # blocking read
            if not line:
                break
            self._queue.put(line)

    def new_lines(self):
        lines = []
        while not self._queue.empty():
            line = self._queue.get()
            lines.append(line.decode(self._text_encoding, errors = 'ignore'))
        self._all_lines.extend(lines)
        return lines

    def all_lines(self):
        self.new_lines()
        return self._all_lines

# ------------------------------------------------------------------------------
#
from chimerax.core.commands import Annotation, AnnotationError, next_token
class LigandsArg(Annotation):
    name = 'ligands'
    allow_repeat = 'expand'

    @classmethod
    def parse(cls, text, session):
        value, used, rest = next_token(text)
        ligands = []
        for ligand in value.split(','):
            count = 1
            if ligand.endswith(')'):
                oi = ligand.rfind('(')
                if oi > 0 :
                    try:
                        count = int(ligand[oi+1:-1])
                        ligand = ligand[:oi]
                    except:
                        pass
            ligands.append((ligand, count))
        return ligands, used, rest

# ------------------------------------------------------------------------------
#
class NamedLigandsArg(Annotation):
    name = 'list of name and smiles string'

    @classmethod
    def parse(cls, text, session):
        value, used, rest = next_token(text)
        names_and_smiles = value.split(',')
        if len(names_and_smiles) % 2 == 1:
            raise AnnotationError('Named ligands must be a comma separated list of names and smiles string, got an odd number of comma-separated values')
        names = names_and_smiles[::2]
        smiles = names_and_smiles[1::2]
        ligands = [(name, smile) for name, smile in zip(names, smiles)]
        return ligands, used, rest

# ------------------------------------------------------------------------------
#
class RepeatSequencesArg(Annotation):
    name = 'sequences'
    allow_repeat = 'expand'

    @classmethod
    def parse(cls, text, session):
        from chimerax.atomic import SequencesArg, Chains
        value, used, rest = SequencesArg.parse(text, session)
        if isinstance(value, Chains):
            value = list(value)		# Require list so repeat args flattens.
        return value, used, rest

# ------------------------------------------------------------------------------
#
def register_boltz_predict_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, SaveFolderNameArg, BoolArg, EnumOf, IntArg, OpenFolderNameArg
    from chimerax.atomic import SequencesArg, ResiduesArg, AtomicStructureArg

    desc = CmdDesc(
        optional = [('sequences', SequencesArg)],
        keyword = [('ligands', ResiduesArg),
                   ('exclude_ligands', StringArg),
                   ('protein', RepeatSequencesArg),
                   ('dna', RepeatSequencesArg),
                   ('rna', RepeatSequencesArg),
                   ('ligand_ccd', LigandsArg),
                   ('ligand_smiles', LigandsArg),
                   ('for_each_smiles_ligand', NamedLigandsArg),
                   ('name', StringArg),
                   ('results_directory', SaveFolderNameArg),
                   ('device', EnumOf(['default', 'cpu', 'gpu'])),
                   ('kernels', BoolArg),
                   ('precision', EnumOf(['32', 'bf16-mixed', '16', 'bf16-true'])),
                   ('float16', BoolArg),
                   ('affinity', StringArg),
                   ('steering', BoolArg),
                   ('samples', IntArg),
                   ('recycles', IntArg),
                   ('seed', IntArg),
                   ('use_msa_cache', BoolArg),
                   ('msa_only', BoolArg),
                   ('open', BoolArg),
                   ('install_location', SaveFolderNameArg),
                   ('wait', BoolArg)],
        synopsis = 'Predict a structure with Boltz',
        url = 'help:boltz_help.html'
    )
    register('boltz predict', desc, boltz_predict, logger=logger)

    desc = CmdDesc(
        required = [('run_directory', OpenFolderNameArg),],
        keyword = [('align_to', AtomicStructureArg),],
        synopsis = 'Show table of Boltz ligand binding prediction results',
        url = 'help:boltz_help.html'
    )
    register('boltz ligandtable', desc, boltz_ligand_table, logger=logger)

