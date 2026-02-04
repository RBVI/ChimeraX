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

def openfold_predict(session, sequences = [], ligands = None, exclude_ligands = 'HOH',
                  protein = [], dna = [], rna = [],
                  ligand_ccd = [], ligand_smiles = [], for_each_smiles_ligand = [],
                  name = None, results_directory = None,
                  device = None, use_server = False, server_host = None, server_port = None,
                  kernels = None, precision = None,
                  samples = 1, recycles = 3, seed = 42,
                  msa_only = False, use_msa_cache = True, msa_cache_dir = '~/Downloads/ChimeraX/OpenFoldMSA',
                  open = True, install_location = None, wait = None):

    if install_location is not None:
        from .settings import _openfold_settings
        settings = _openfold_settings(session)
        settings.openfold_install_location = install_location
        settings.save()

    if not _is_openfold_available(session):
        return None

    if wait is None:
        wait = False if session.ui.is_gui else True

    polymer_components, modeled_chains, unmodeled_chains = _polymer_components(sequences, protein, dna, rna)
    align_to = modeled_chains[0].structure if modeled_chains else None
    used_chain_ids = set(sum((pc.chain_ids for pc in polymer_components), []))
    ligand_components, covalent_ligands = _ligand_components(ligands, exclude_ligands.split(','),
                                                             ligand_ccd, ligand_smiles, used_chain_ids)
    molecular_components = polymer_components + ligand_components

    if len(molecular_components) == 0 and len(for_each_smiles_ligand) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No molecules specified for OpenFold prediction')
   
    # Warn about unmodeled compnents
    if unmodeled_chains:
        msg = f'Chains {", ".join(unmodeled_chains)} not modeled because not protein/DNA/RNA'
        session.logger.warning(msg)
    if covalent_ligands:
        session.logger.info(f'Predicting covalent ligands not yet supported: {covalent_ligands}')

    for_each_ligand = _for_each_ligand(for_each_smiles_ligand, used_chain_ids, session.logger)
    if for_each_ligand is None:
        p = OpenFoldPrediction(name, molecular_components, align_to = align_to)
        predictions = [p]
    else:
        predictions = _each_ligand_predictions(for_each_ligand, molecular_components, align_to)
            
    br = OpenFoldRun(session, predictions, name = name, run_directory = results_directory,
                  samples = samples, recycles = recycles, seed = seed,
                  device = device, use_server = use_server, server_host = server_host, server_port = server_port,
                  use_kernels = kernels, precision = precision,
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
            polymer_type = _chain_type(seq) if is_chain else type  # protein, dna, rna or None
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

    # Create OpenFoldMolecules
    polymer_components = [OpenFoldMolecule(polymer_type, chain_ids, sequence_string = seq_string)
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
        ligand_components.append(OpenFoldMolecule('ligand', chain_ids, ccd_code = ccd))

    for smiles, count in ligand_smiles:
        chain_ids = [_next_chain_id(used_chain_ids) for i in range(count)]
        ligand_components.append(OpenFoldMolecule('ligand', chain_ids, smiles_string = smiles))
        
    return ligand_components, covalent_ligands

# ------------------------------------------------------------------------------
#
def _for_each_ligand(name_and_smiles, used_chain_ids, log):
    n = len(name_and_smiles)
    if n == 0:
        return None

    chain_ids = [_next_chain_id(used_chain_ids)]
    ligands = [OpenFoldMolecule('ligand', smiles_string = smiles, name = name.replace(' ', '_'), chain_ids = chain_ids)
               for name, smiles in name_and_smiles]
    return ligands
        
# ------------------------------------------------------------------------------
#
def _each_ligand_predictions(for_each_ligand, molecular_components, align_to):
    predictions = []
    for ligand in for_each_ligand:
        components = molecular_components + [ligand]
        p = OpenFoldPrediction(ligand.name, components, align_to = align_to)
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
def _msa_run(session, name, molecular_components, msa_cache_dir, wait):
    proteins = [mc for mc in molecular_components if mc.type == 'protein']
    protein_seqs = [mc.sequence_string for mc in proteins]
    if protein_seqs:
        msa_cache_files = _find_msa_cache_files(protein_seqs, msa_cache_dir)
        if not msa_cache_files:
            msa_name = f'{name}_msa' if name else 'msa'
            prediction = OpenFoldPrediction(msa_name, proteins)
            br = OpenFoldRun(session, prediction, msa_only = True, msa_cache_dir = msa_cache_dir,
                          wait = wait)
            return br
    return None

# ------------------------------------------------------------------------------
#
def _start_multiple_runs(openfold_runs, wait = False):
    # Run MSA server calculation before multiple ligand structure predictions.
    if wait:
        for br in openfold_runs:
            br.start(wait = True)
            if not br.success:
                break
    else:
        def run_next(openfold_runs, previous_run = None):
            if openfold_runs and (previous_run is None or previous_run.success):
                openfold_runs[0].start(finished_callback =
                                    lambda: run_next(openfold_runs[1:], previous_run = openfold_runs[0]))
        run_next(openfold_runs)

# ------------------------------------------------------------------------------
#
class OpenFoldPrediction:
    '''Set of molecular components for predicting a single structure with OpenFold.'''
    def __init__(self, name, molecular_components, align_to = None):
        self.name = name
        self._molecular_components = molecular_components  # List of OpenFoldMolecule
        self._align_to = align_to	                   # AtomicStructure to align prediction to.

    def json_filename(self, default_name = 'input'):
        if self.name is None:
            self.name = default_name
        return self.name + '.json'

    def input(self, msa_cache_directory = None, msa_directory = None, msa_relative_to_path = None):

        # Create yaml for polymers
        msa_files = self._msa_cache_files(msa_cache_directory)
        if msa_directory:
            msa_files = self._copy_msa_files(msa_files, msa_directory, msa_relative_to_path)

        components = []
        for mc in self._molecular_components:
            if mc.type in ('protein', 'dna', 'rna'):
                component ={"molecule_type": mc.type,
                            "chain_ids": mc.chain_ids,
                            "sequence": mc.sequence_string}
                if msa_files and mc.type == 'protein':
                    msa = msa_files.paths[mc.sequence_string]
                    if msa['unpaired']:
                        component["main_msa_file_paths"] = [msa['unpaired']]
                    if msa['paired']:
                        component["paired_msa_file_paths"] = [msa['paired']]
                    if msa['templates']:
                        component["template_alignment_file_path"] = msa['templates']
                        component["template_entry_chain_ids"] = msa['template_ids']
                components.append(component)

        # Create json for ligands
        for mc in self._molecular_components:
            if mc.type == 'ligand':
                component = {"molecule_type": "ligand",
                             "chain_ids": mc.chain_ids}
                if mc.ccd_code:
                    component["ccd_codes"] = [mc.ccd_code]
                elif mc.smiles_string:
                    smiles = mc.smiles_string.replace('\\', '\\\\')  # Escape backslashes
                    component["smiles"] = smiles
                components.append(component)

        input = { f"{self.name}": { "chains": components }}

        msa_dir = msa_files.directory if msa_files else '.'
        msa_path_yaml = f'''msa_computation_settings:
  msa_output_directory: {msa_dir}/colabfold_msas
  cleanup_msa_dir: False
  save_mappings: True 

template_preprocessor_settings:
  output_directory: {msa_dir}/colabfold_templates
'''
        return input, msa_path_yaml

    def _msa_cache_files(self, msa_cache_directory):
        if msa_cache_directory:
            protein_seqs = [mc.sequence_string for mc in self._molecular_components if mc.type == 'protein']
            msa_cache_files = _find_msa_cache_files(protein_seqs, msa_cache_directory)
        else:
            msa_cache_files = None

        self.cached_msa_dir = msa_cache_files.directory if msa_cache_files else None

        return msa_cache_files

    def _copy_msa_files(self, msa_files, msa_directory, msa_relative_to_path):
        # Copy MSA files to run directory so they are sent to server.
        from shutil import copy2
        for path in set(msa_files.values()):
            copy2(path, msa_directory)
        from os.path import join, basename, relpath
        copied_msa_files = {seq:join(msa_directory, basename(msa_path)) for seq, msa_path in msa_files.items()}
        if msa_relative_to_path:
            copied_msa_files = {seq:relpath(msa_path, msa_relative_to_path)
                                for seq, msa_path in copied_msa_files.items()}
        return copied_msa_files
    
    @property
    def using_cached_msa(self):
        return self.cached_msa_dir is not None
    
    def _add_to_msa_cache(self, msa_directory, template_directory, cache_directory):
        if self.cached_msa_dir:
            return False
        protein_seqs = [mc.sequence_string for mc in self._molecular_components if mc.type == 'protein']
        if len(protein_seqs) == 0:
            return False
        from os.path import exists, join
        if not exists(msa_directory) and not exists(template_directory):
            return False
        if self.name is None:
            return False
        return _add_to_msa_cache(self.name, protein_seqs, msa_directory, template_directory, cache_directory)

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
    
    def open_predictions(self, session, mmcif_paths, align = True, color = True):
        models = []
        for mmcif_path in mmcif_paths:
            pmodels = self._open_prediction(session, mmcif_path, align=align, color=color)
            models.extend(pmodels)
        return models

    def _open_prediction(self, session, mmcif_path, align = True, color = True):
        # Find path to predicted model
        from os.path import exists
        if not exists(mmcif_path):
            session.logger.warning('Prediction file not found: %s' % mmcif_path)
            return []

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
        
# ------------------------------------------------------------------------------
#
class OpenFoldMolecule:
    def __init__(self, type, chain_ids, sequence_string = None, ccd_code = None, smiles_string = None, name = None):
        self.type = type	# protein, dna, rna, ligand
        self.chain_ids = chain_ids
        self.sequence_string = sequence_string
        self.ccd_code = ccd_code
        self.smiles_string = smiles_string
        self.name = name

# ------------------------------------------------------------------------------
#
class OpenFoldRun:
    def __init__(self, session, structures, name = None, run_directory = None,
                 samples = 1, recycles = 3, seed = 42,
                 device = 'default', use_server = False, server_host = None, server_port = None,
                 use_kernels = None, precision = None,
                 msa_only = False, use_msa_cache = True, msa_cache_dir = '~/Downloads/ChimeraX/OpenFoldMSA',
                 open = True, wait = False):

        self._session = session
        self._predictions = [structures] if isinstance(structures, OpenFoldPrediction) else structures
        self.name = name
        self._samples = samples		# Number of predicted structures
        self._recycles = recycles	# Number of openfold recycling steps
        self._device = device		# gpu, cpu or default, or None (uses settings value)
        self._use_server = use_server	# True or False
        if use_server:
            if server_host is None:
                server_host = self._settings.server_host
            if server_port is None:
                server_port = self._settings.server_port
        self._server_host = server_host # Host name, e.g. minsky.cgl.ucsf.edu
        self._server_port = server_port # Port number
        self._use_kernels = use_kernels	# whether to use cuequivariance module for triangle attention
        self._precision = precision	# "32", "bf16-mixed", "16", "bf16-true"
        self._seed = seed		# Random seed for computation
        self._open = open		# Whether to open predictions when openfold finishes.

        from os.path import abspath, isabs
        run_dir = abspath(run_directory) if run_directory and not isabs(run_directory) else run_directory
        self._run_directory = run_dir	# Location of input and results files
        self._input_path = None		# json file path or directory of json files
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
        self._write_json_input_files()
        self._write_ligands_file()
        self._run_openfold()

    @property
    def _settings(self):
        from .settings import _openfold_settings
        settings = _openfold_settings(self._session)
        return settings

    def _write_json_input_files(self):
        # Create json before making directory so directory is not created if json creation fails.
        self._run_directory = dir = self._unique_run_directory()

        if self.name is None:
            from os.path import basename
            self.name = basename(dir)

        msa_cache_dir = self._msa_cache_dir if self._use_msa_cache else None
        msa_directory = msa_relative_to_path = (self._run_directory if self._use_server else None)
        queries = {}
        for p in self._predictions:
            query, msa_path_yaml = p.input(msa_cache_dir, msa_directory, msa_relative_to_path)
            queries.update(query)
        input = {'queries' : queries}
        import json
        json_text = json.dumps(input, indent=2)
        from os.path import join
        json_path = join(dir, f'{self.name}.json')
        with open(json_path, 'w') as f:
            f.write(json_text)

        # TODO: All predictions have to have the same MSA directory.  Verify this.
        runner_yaml_path = join(dir, 'msa_path.yaml')
        with open(runner_yaml_path, 'w') as f:
            f.write(msa_path_yaml)
                
        self._input_path = json_path
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
            dir = self._settings.openfold_results_location

        from os.path import expanduser
        dir = expanduser(dir)

        rdir = self._add_directory_suffix(dir)

        from os.path import exists
        if not exists(rdir):
            from os import makedirs
            makedirs(rdir)
        else:
            msg = f'OpenFold prediction directory {rdir} already exists.  Files will be overwritten.'
            self._session.logger.warning(msg)

        return rdir

    def _add_directory_suffix(self, dir):
        if '[N]' not in dir and '[name]' not in dir:
            return dir
            
        if self.name:
            dir = dir.replace('[name]', self.name)  # Handle old openfold tool that used [N] for the name.
            dir = dir.replace('[N]', self.name)  # Handle old openfold tool that used [N] for the name.
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
            device = openfold_default_device(self._session)
        else:
            device = self._device
        return device

    def _run_openfold(self):
        self._running = True
        msg = f'sending to server {self._server_host}' if self._use_server else 'starting OpenFold'
        self._set_stage(msg)

        self._log_prediction_info()

        command = self._prediction_command()
        self._write_command_file(command)

        from time import time
        self._start_time = time()

        if self._use_server:
            self._run_on_server()
        else:
            self._start_subprocess(command)

    def _start_subprocess(self, command):
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

        from subprocess import Popen, PIPE
        from .install import _no_subprocess_window
        # To continue to run even if ChimeraX exits use start_new_session=True
        p = Popen(command, cwd = self._run_directory,
                  stdout = PIPE, stderr = PIPE, env=env,
                  creationflags = _no_subprocess_window())
        self._process = p

        self._monitor_openfold_output()
        
        if self._wait:
            self._check_process_completion()
        else:
            self._session.triggers.add_handler('new frame', self._check_process_completion)

    def _run_on_server(self):
        '''This blocks until the prediction is finished.'''
        run_dir = self._run_directory
        from .server import predict_on_server
        job_id = predict_on_server(run_dir, self._server_host, self._server_port)

        active_jobs = list(self._settings.active_server_jobs)
        active_jobs.append(run_dir)
        self._settings.active_server_jobs = active_jobs

        class WaitForServerPrediction:
            def __init__(self, prediction, job_id, run_dir, host, port, check_interval = 10):
                self._prediction = prediction
                self._job_id = job_id
                self._run_dir = run_dir
                self._server_host = host
                self._server_port = port
                self._check_interval = check_interval
                from time import time
                self._next_check_time = time() + check_interval
                triggers = prediction._session.triggers
                triggers.add_handler('new frame', self._check_for_server_results)
            def _check_for_server_results(self, trigger_name, trigger_data):
                from time import time
                if time() < self._next_check_time:
                    return
                self._next_check_time = time() + self._check_interval
                
                from .server import get_results
                msg = get_results(self._job_id, self._run_dir, self._server_host, self._server_port)
                logger = self._prediction._session.logger
                if msg == 'Done':
                    self._prediction._server_job_finished(self._run_dir)
                    return 'delete handler'
                elif msg == 'No such job':
                    logger.error(f'OpenFold server could not find job {self._job_id}')
                    self._prediction._prediction_finished(success = False)
                    return 'delete handler'
                elif msg.startswith('Error'):
                    logger.bug(f'OpenFold server error for job {self._job_id}: {msg}')
                    self._prediction._prediction_finished(success = False)
                    return 'delete handler'
                else:
                    status = f'{msg} {self._job_id} on {self._server_host}'
                    self._prediction._set_stage(status)
            
        WaitForServerPrediction(self, job_id, run_dir, self._server_host, self._server_port)

    def _server_job_finished(self, run_dir):
        from os.path import join
        with open(join(run_dir, 'stdout'), 'r', encoding = 'utf-8') as f:
            stdout = f.read()
        with open(join(run_dir, 'stderr'), 'r', encoding = 'utf-8') as f:
            stderr = f.read()

        jobs = self._settings.active_server_jobs
        if run_dir in jobs:
            jobs.remove(run_dir)
            self._settings.active_server_jobs = list(jobs)

        struct_files = self._prediction_cif_files()
        exit_code = 0 if struct_files else 1
        self._process_completed(exit_code, stdout, stderr)

    def _prediction_cif_files(self):
        struct_files = []
        from os import listdir
        from os.path import isdir, join
        for pdir in self._prediction_directories:
            if isdir(pdir):
                struct_files.extend(join(pdir,filename) for filename in listdir(pdir)
                                    if filename.endswith('.cif'))
        return struct_files
        
    def _prediction_command(self):
        openfold_venv = self._settings.openfold_install_location
        from .install import find_executable
        openfold_exe = find_executable(openfold_venv, 'run_openfold')

        command = [openfold_exe, 'predict']

        # Input file
        command.append(f'--query_json={self._input_path}')

        # Save MSAs and templates in the prediction directory or read from cache
        command.append('--runner_yaml=msa_path.yaml')

        if self._msa_only:
            command.append('--msa_and_templates_only=true')
            return command

        if not self._use_msa_server:
            command.append('--use_msa_server=false')
        
        if self._samples != 5:
            command.append(f'--num_diffusion_samples={self._samples}')

        if self._seed is not None and self._seed != 42:
            command.append(f'--seed={self._seed}')

        '''
        command.extend(['--accelerator', self.device])

        use_kernels = self._use_kernels
        if self._use_kernels is None:
            from sys import platform
            use_kernels = (self.device == 'gpu' and platform == 'linux')
        if not use_kernels:
            command.append('--no_kernels')

        if self._precision is not None:
            command.extend(['--precision', self._precision])
            
        if self._recycles != 3:
            command.extend(['--recycling_steps', str(self._recycles)])
        '''

        return command

    def _write_command_file(self, command):
        # Save command to a file
        from os.path import join
        command_file = join(self._run_directory, 'command')
        self._command = cmd = ' '.join(command)
        with open(command_file, 'w') as f:
            f.write(cmd)
    
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
        log.info(f'Running OpenFold prediction of {mol_descrip} on {device}')

        if self._use_msa_server:
            msa_method = 'Using multiple sequence alignment server https://api.colabfold.com'
        elif len(pred) == 1:
            msa_method = f'Using cached multiple sequence alignment {pred[0].cached_msa_dir}'
        else:
            msa_method = f'Using cached multiple sequence alignment'
        log.info(msa_method)

    def _monitor_openfold_output(self):
        p = self._process
        self._stdout = ReadOutputThread(p.stdout)
        self._stderr = ReadOutputThread(p.stderr)

    def _check_process_completion(self, *trigger_args):
        p = self._process
        if self._wait:
            while p.poll() is None:
                self._check_openfold_output()
                from time import sleep
                sleep(1)
        else:
            self._check_openfold_output()
            if p.poll() is None:
                return		    # Process still running

        self._set_stage('')	    # Job finished

        stdout = ''.join(self._stdout.all_lines())
        stderr = ''.join(self._stderr.all_lines())
        self._save_stdout_stderr(stdout, stderr)

        self._process_completed(p.returncode, stdout, stderr)

        return 'delete handler'

    def _check_openfold_output(self):

        new_lines = self._stderr.new_lines()
        if len(new_lines) == 0:
            return

        if len(self._predictions) > 1:
            # Report ligand being docked in status messages.
            detail_func = lambda text: text[25:].strip()
        else:
            detail_func = lambda text: ''
            
        stages = [('Loading weights', 'loading weights'),
                  ('Finished loading weights', ''),
                  ('Loading model state', 'initializing neural net'),
                  ('Finished loading model state', ''),
                  ('SUBMIT', 'submitting sequence search'),
                  ('RATELIMIT', 'sequence server busy... waiting'),
                  ('PENDING', 'sequence search submitted'),
                  ('RUNNING', 'sequence search running'),
                  ('COMPLETE', 'sequence search finished'),
                  ('Preprocessing templates', 'processing templates'),
                  ('Finished preprocessing templates', ''),
                  ('Creating features', 'creating neural net input'),
                  ('Finished creating features', ''),
                  ('Started inference', 'structure inference'),
                  ('Computing confidence scores', 'computing confidence'),
                  ('Finished computing confidence scores', ''),
                  ('Beginning structure inference ', ('structure inference', detail_func)),
                  ('Finished prediction inference ', ''),
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

    def _save_stdout_stderr(self, stdout, stderr):
        dir = self._run_directory
        from os.path import join
        with open(join(dir, 'stdout'), 'w', encoding = 'utf-8') as f:
            f.write(stdout)
        with open(join(dir, 'stderr'), 'w', encoding = 'utf-8') as f:
            f.write(stderr)
    
    def _process_completed(self, exit_code, stdout, stderr):

        msg = self._prediction_failed_message(exit_code, stdout, stderr)
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
                self._report_confidence(self._predictions[0])
                self._report_runtime()
                self._cite()
                if self._open:
                    models = []
                    for p in self._predictions:
                        mmcif_paths = [self._mmcif_path(p.name, i+1) for i in range(self._samples)]
                        models.extend(p.open_predictions(self._session, mmcif_paths))
                    self._opened_predictions = models
            success = True

        for p in self._predictions:
            p._add_to_msa_cache(self._msa_directory, self._template_directory, self._msa_cache_dir)

        self._prediction_finished(success)

    def _prediction_finished(self, success):
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
        sut = st.get('starting OpenFold', 0)
        parts.append(f'start openfold {"%.0f" % sut} sec')
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

        timings = ', '.join(parts)
        msg = f'OpenFold prediction completed in {"%.0f" % total} seconds ({timings})'
        self._session.logger.info(msg)

        if self._use_msa_server and wait_t >= 60:
            msg = f'The sequence alignment server api.colabfold.com was busy and OpenFold waited {"%.0f" % wait_t} seconds to start the alignment computation for this prediction.'
            self._session.logger.warning(msg)
            
    def _prediction_failed_message(self, exit_code, stdout, stderr):

        msg = None
        if self._prediction_ran_out_of_memory(stdout):
            msg = ('The OpenFold prediction ran out of memory.  The memory use depends on the'
                   ' number of protein and nucleic acid residues plus the number of ligand'
                   ' atoms.  You can reduce the size of your molecular assembly to stay'
                   ' within the memory limits.')
        elif exit_code != 0 or (not self._msa_only and not self._predicted_model_exists()):
            if 'No supported gpu backend found' in stderr:
                msg = ('Attempted to run OpenFold on the GPU but no supported GPU device could be found.'
                       ' To avoid this error specify the compute device as "cpu" in the ChimeraX OpenFold'
                       ' options panel, or using the ChimeraX openfold command "device cpu" option.'
                       ' OpenFold supports Nvidia GPUs with CUDA and Mac M series GPUs.  On Windows the OpenFold'
                       ' installation installs torch with no GPU support, so using Nvidia GPUs on'
                       ' Windows requires reinstalling gpu-enabled torch with OpenFold which we plan to'
                       ' support in the future.')
            elif 'ValueError: CCD component ' in stderr:
                i_ccd_start = stderr.find('ValueError: CCD component ') + 26
                i_ccd_end = i_ccd_start + stderr[i_ccd_start:].find(' ')
                ccd_code = stderr[i_ccd_start:i_ccd_end]
                msg = ('Your OpenFold installation does not have a molecular structure for'
                       f' PDB chemical component dictionary code {ccd_code} either because'
                       ' that code is new or is mistyped.  You can try specifying that ligand'
                       f' using a SMILES string from https://www.rcsb.org/ligand/{ccd_code}'
                       ' instead of using its CCD code.')
            elif 'load_from_checkpoint' in stderr and 'PytorchStreamReader failed reading zip archive' in stderr:
                msg = ('OpenFold failed reading neural network weights from directory ~/.openfold.'
                       ' This can happen if you quit ChimeraX while installing OpenFold before the'
                       ' installation finished. To fix it delete the ~/.openfold directory.'
                       ' The next time a OpenFold prediction is run it will download the neural'
                       ' network weights and chemical components (5 Gbytes). That may take'
                       ' minutes to hours depending on your internet connection speed.')
            else:
                if self._user_terminated:
                    msg = 'Prediction terminated by user'
                    self._user_terminated = False
                else:
                    msg = '\n'.join([
                        f'Running openfold prediction failed with exit code {exit_code}:',
                        'command:',self._command,
                        'stdout:', stdout,
                        'stderr:', stderr,
                        ])
        return msg

    def _predicted_model_exists(self):
        from os.path import join, exists
        for prediction in self._predictions:
            mmcif_path = self._mmcif_path(prediction.name)
            if exists(mmcif_path):
                return True
            print ('did not find prediction at', mmcif_path)
        return False
        
    @property
    def finished(self):
        return self._finished

    def terminate(self):
        if self._running:
            self._process.kill()
            self._user_terminated = True

    def _prediction_ran_out_of_memory(self, stdout):
        return len(self._prediction_cif_files()) == 0 and 'ran out of memory' in stdout

    def _mmcif_path(self, prediction_name, sample=1):
        from os.path import join
        return join(self._run_directory, prediction_name, f'seed_{self._seed}',
                    f'{prediction_name}_seed_{self._seed}_sample_{sample}_model.cif')

    def _confidence_path(self, prediction_name, sample=1):
        from os.path import join
        return join(self._run_directory, prediction_name, f'seed_{self._seed}',
                    f'{prediction_name}_seed_{self._seed}_sample_{sample}_confidences_aggregated.json')

    @property
    def _results_directory(self):
        from pathlib import Path
        name = Path(self._input_path).stem
        from os.path import join
        return join(self._run_directory, f'openfold_results_{name}')

    @property
    def _prediction_directories(self):
        from os.path import join
        return [join(self._run_directory, p.name, f'seed_{self._seed}') for p in self._predictions]

    @property
    def _msa_directory(self):
        from os.path import join
        return join(self._run_directory, 'colabfold_msas')

    @property
    def _template_directory(self):
        from os.path import join
        return join(self._run_directory, 'colabfold_templates')

    def _report_confidence(self, prediction):
        from os.path import join, exists
        lines = []
        for sample in range(self._samples):
            conf_path = self._confidence_path(prediction.name, sample+1)
            results = _read_json(conf_path)
            if not results:
                continue
            ptm = results.get('ptm', -1)
            iptm = results.get('iptm', -1)
            iptm_text = f'ipTM {"%.2f" % iptm}' if iptm != 0 else ''
            plddt = results.get('avg_plddt', -1)
            parts = [f'Confidence pTM {"%.2f" % ptm}',
                     iptm_text,
                     f'pLDDT {"%.0f" % plddt}']
            lines.append(', '.join(parts))
        if lines:
            self._session.logger.info('<br>'.join(lines), is_html = True)

    def _report_multi_prediction_results(self):
        n = len(self._predictions)
        msg = f'Predicted structures for {n} ligands'
        self._session.logger.info(msg)
        self._report_runtime()
        # Create table of results
        if self._session.ui.is_gui:
            align_to = self._predictions[0]._align_to if self._predictions else None
            from .openfold_gui import LigandPredictionsTable
            t = LigandPredictionsTable(self._session, self._input_path, align_to = align_to)
            self._save_results_as_csv_file(t)
            # TODO: Would be nice to save csv file even if no gui is available.

    def _save_results_as_csv_file(self, table):
        from os.path import join
        csv_path = join(self._run_directory, f'{self.name}.oflig')
        table.save_csv_file(csv_path)
        from chimerax.core.filehistory import remember_file
        remember_file(self._session, csv_path, 'oflig', models=[], file_saved=True)

    def _cite(self):
        session = self._session
        if not hasattr(session, '_cite_openfold'):
            msg = 'Please cite <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC11601547/">OpenFold-1 Democratizing Biomolecular Interaction Modeling. BioRxiv https://doi.org/10.1101/2024.11.19.624167</a> if you use these predictions.'
            session.logger.info(msg, is_html = True)
            session._cite_openfold = True  # Only log this message once per session.

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
def openfold_ligand_table(session, run_directory, include_smiles = True, align_to = None):
    '''Show a table of OpenFold ligand binding prediction results.'''
    from os.path import join, basename, exists
    from os import listdir
    dir_name = basename(run_directory)
    query_json = [filename for filename in listdir(run_directory)
                  if filename.endswith('.json') and dir_name.startswith(filename[:-5])]
    if len(query_json) != 1:
        from chimerax.core.errors import UserError
        raise UserError(f'Did not find .json query file in {run_directory} matching directory name.')
    query_path = join(run_directory, query_json[0])
    from .openfold_gui import LigandPredictionsTable
    LigandPredictionsTable(session, query_path, align_to = align_to)

# ------------------------------------------------------------------------------
#
def _is_openfold_available(session):
    '''Check if OpenFold is locally installed with paths properly setup.'''
    from .settings import _openfold_settings
    settings = _openfold_settings(session)
    install_location = settings.openfold_install_location
    from os.path import isdir
    if not isdir(install_location):
        msg = 'You need to install OpenFold by pressing the "Install OpenFold" button on the ChimeraX OpenFold user interface, or using the ChimeraX command "openfold install".  If you already have OpenFold installed, you can set the OpenFold installation location in the user interface under Options, or use the installLocation option of the ChimeraX openfold command "openfold predict ... installLocation /path/to/openfold"'
        session.logger.error(msg)
        return False

    if not _check_venv_valid(session, install_location):
        return False

    return True

# ------------------------------------------------------------------------------
#
def _check_venv_valid(session, install_location):
    '''
    Warn if the ChimeraX python that was used for openfold venv does not exist.
    '''
    from os.path import join, exists
    venv_config_path = join(install_location, 'pyvenv.cfg')
    if not exists(venv_config_path):
        # Maybe the user installed OpenFold themselves.  Don't complain in that case.
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
        msg = f'The ChimeraX version you used to install OpenFold no longer exists and OpenFold uses the Python from that ChimeraX.  You need to reinstall OpenFold with your current ChimeraX.  First remove the directory containing the old OpenFold installation\n\n{install_location}\n\nThen restart ChimeraX and press the "Install OpenFold" button on the ChimeraX OpenFold panel or use the ChimeraX command\n\nopenfold install {install_loc}'
        session.logger.error(msg)
        return False

    return True

# ------------------------------------------------------------------------------
#
def openfold_default_device(session):
    from sys import platform
    if platform in ('win32', 'linux'):
        from .install import have_nvidia_driver
        device = 'gpu' if have_nvidia_driver() and _torch_has_cuda(session) else 'cpu'
        # TODO: On Linux run nvidia-smi to see if GPU memory is sufficient to run OpenFold.
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
    from .settings import _openfold_settings
    settings = _openfold_settings(session)
    openfold_install = settings.openfold_install_location
    from os.path import join, exists
    torch_cuda_lib = join(openfold_install, lib_path)
    return exists(torch_cuda_lib)

# ------------------------------------------------------------------------------
#
def _find_msa_cache_files(protein_seqs, msa_cache_dir):
    from os.path import exists, join, expanduser
    msa_cache_dir = expanduser(msa_cache_dir)
    index_path = join(msa_cache_dir, 'index')
    if not exists(msa_cache_dir) or not exists(index_path):
        return None
    with open(index_path, 'r') as f:
        index_lines = f.readlines()
    for line in index_lines:
        fields = line.strip().split(',')
        msa_dir, msa_seqs = fields[0], fields[1:]
        if msa_seqs == protein_seqs:
            msa_dir = join(msa_cache_dir, msa_dir)
            return MSACacheFiles(msa_seqs, msa_dir)
    return None

# ------------------------------------------------------------------------------
#
class MSACacheFiles:
    def __init__(self, msa_seqs, msa_dir, suffix = '.npz'):
        self.sequences = msa_seqs
        self.directory = msa_dir
        self.file_suffix = suffix
        self.paths = self._openfold_msa_paths()

    def _openfold_msa_paths(self):
        msa_paths = {seq: {'unpaired':None, 'paired':None, 'templates':None, 'template_ids':None}
                     for seq in self.sequences}

        from os.path import join, exists
        msa_directory = join(self.directory, 'colabfold_msas')
        template_directory = join(self.directory, 'colabfold_templates')

        # Map sequence number to filename (sha256 hash of sequence).
        seq_ids_path = join(msa_directory, 'mappings', 'seq_to_rep_id.json')
        if not exists(seq_ids_path):
            return msa_paths
        import json
        with open(seq_ids_path, 'r') as f:
            seq_ids = json.load(f)

        # Find unpaired MSAs
        for seq, id in seq_ids.items():
            unpaired_path = join(msa_directory, 'main', id + '.npz')
            if exists(unpaired_path):
                msa_paths[seq]['unpaired'] = unpaired_path

        # Find paired MSAs
        complex_ids_path = join(msa_directory, 'mappings', 'query_name_to_complex_id.json')
        if exists(complex_ids_path):
            with open(complex_ids_path, 'r') as f:
                complex_ids = json.load(f)
            if len(complex_ids) == 1:
                complex_id = tuple(complex_ids.values())[0]
                for seq, id in seq_ids.items():
                    paired_path = join(msa_directory, 'paired', complex_id, id + '.npz')
                    if exists(paired_path):
                        msa_paths[seq]['paired'] = paired_path

        # Find template alignment files
        for seq, id in seq_ids.items():
            template_path = join(template_directory, 'template_cache', id + '.npz')
            if exists(template_path):
                msa_paths[seq]['templates'] = template_path
                import numpy
                with numpy.load(template_path) as data:
                    msa_paths[seq]['template_ids'] = tuple(data.files)

        return msa_paths

# ------------------------------------------------------------------------------
#
def _add_to_msa_cache(dir_name, protein_seqs, msa_directory, template_directory, msa_cache_dir):

    from os.path import exists, join, basename
    if not exists(msa_cache_dir):
        from os import makedirs
        makedirs(msa_cache_dir)

    new_cache_dir = join(msa_cache_dir, dir_name)
    if exists(new_cache_dir):
        new_cache_dir = _unique_cache_dir(new_cache_dir)
        dir_name = basename(new_cache_dir)

    from shutil import copytree
    copytree(msa_directory, join(new_cache_dir, basename(msa_directory)))
    copytree(template_directory, join(new_cache_dir, basename(template_directory)), dirs_exist_ok = True)

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
def register_openfold_predict_command(logger):
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
                   ('use_server', BoolArg),
                   ('server_host', StringArg),
                   ('server_port', IntArg),
                   ('kernels', BoolArg),
                   ('precision', EnumOf(['32', 'bf16-mixed', '16', 'bf16-true'])),
                   ('samples', IntArg),
                   ('recycles', IntArg),
                   ('seed', IntArg),
                   ('use_msa_cache', BoolArg),
                   ('msa_only', BoolArg),
                   ('open', BoolArg),
                   ('install_location', SaveFolderNameArg),
                   ('wait', BoolArg)],
        synopsis = 'Predict a structure with OpenFold',
        url = 'https://www.rbvi.ucsf.edu/chimerax/data/openfold-feb2026/openfold_help.html'
    )
    register('openfold predict', desc, openfold_predict, logger=logger)

    desc = CmdDesc(
        required = [('run_directory', OpenFolderNameArg),],
        keyword = [('include_smiles', BoolArg),
                   ('align_to', AtomicStructureArg),],
        synopsis = 'Show table of OpenFold ligand binding prediction results',
        url = 'https://www.rbvi.ucsf.edu/chimerax/data/openfold-feb2026/openfold_help.html#ligandtablecommand'
    )
    register('openfold ligandtable', desc, openfold_ligand_table, logger=logger)

