# ================================================================================================
# Google Colab code for running an AlphaFold structure prediction.
#

# Make sure virtual machine has a GPU
def check_for_gpu():
    import jax
    devtype = jax.local_devices()[0].platform
    if devtype == 'gpu':
        print ('Have Colab GPU runtime')
    else:
        raise RuntimeError('Require Colab GPU runtime, got %s.\n' % devtype +
                           'Change GPU with Colab menu\n' +
                           'Runtime -> Change Runtime Type -> Hardware accelerator -> GPU.')

def is_alphafold_installed():
    try:
        import alphafold
    except:
        return False
    return True

def install_alphafold(ALPHAFOLD_GIT_REPO = 'https://github.com/deepmind/alphafold',
                      ALPHAFOLD_PARAMETERS = 'https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar',
                      BOND_PARAMETERS = 'https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt',
                      INSTALL_LOG = 'install_log.txt'):

    PARAMS_DIR = './alphafold/data/params'
    import os.path
    PARAMS_PATH = os.path.join(PARAMS_DIR, os.path.basename(ALPHAFOLD_PARAMETERS))

    cmds = f'''
# Get AlphaFold from GitHub and install it
git clone {ALPHAFOLD_GIT_REPO} alphafold
pip3 install ./alphafold

# Get AlphaFold parameters, 3.5 Gbytes,
# Ten models model_1, model_2, ..., model_5, model_1_ptm, ..., model_5_ptm.
mkdir -p "{PARAMS_DIR}"
wget -q -O "{PARAMS_PATH}" {ALPHAFOLD_PARAMETERS}
tar --extract --verbose --file="{PARAMS_PATH}" --directory="{PARAMS_DIR}" --preserve-permissions
rm "{PARAMS_PATH}"

# Get standard bond length and bond angle parameters
mkdir -p /content/alphafold/common
wget -q -P /content/alphafold/common {BOND_PARAMETERS}

# Create a ramdisk to store a database chunk to make jackhmmer run fast.
# Module alphafold.data.tools.jackhmmer makes use of this /tmp/ramdisk.
sudo mkdir -m 777 --parents /tmp/ramdisk
sudo mount -t tmpfs -o size=9G ramdisk /tmp/ramdisk
'''
    run_shell_commands(cmds, 'install_alphafold.sh', INSTALL_LOG)
    
def run_shell_commands(commands, filename, INSTALL_LOG):
    with open(filename, 'w') as f:
        f.write(commands)

    # The -x option logs each command with a prompt in front of it.
    !bash -x "{filename}" >> "{INSTALL_LOG}" 2>&1
    
def install_hmmer(INSTALL_LOG = 'install_log.txt'):
    # Install HMMER package in /usr/bin
    cmds = '''sudo apt install --quiet --yes hmmer'''
    run_shell_commands(cmds, 'install_hmmer.sh', INSTALL_LOG)
    
def install_openmm(CONDA_INSTALL_SH = 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh',
                   INSTALL_LOG = 'install_log.txt'):
    '''Must install alphafold first since an openmm patch from alphafold is used.'''
    # Install Conda
    import os.path
    CONDA_INSTALL = os.path.join('/tmp', os.path.basename(CONDA_INSTALL_SH))
    cmds = f'''
wget -q -P /tmp {CONDA_INSTALL_SH} \
    && bash "{CONDA_INSTALL}" -b -p /opt/conda \
    && rm "{CONDA_INSTALL}"

# Install Python, OpenMM and pdbfixer in Conda
/opt/conda/bin/conda update -qy conda && \
    /opt/conda/bin/conda install -qy -c conda-forge python=3.7 openmm=7.5.1 pdbfixer


# Patch OpenMM
(cd /opt/conda/lib/python3.7/site-packages/ && \
    patch -p0 < /content/alphafold/docker/openmm.patch)

# Put OpenMM and pdbfixer in ipython path which includes current directory /content
ln -s /opt/conda/lib/python3.7/site-packages/simtk .
ln -s /opt/conda/lib/python3.7/site-packages/pdbfixer .
'''
    run_shell_commands(cmds, 'install_openmm.sh', INSTALL_LOG)

# ================================================================================================
# Python routines to run a prediction.
#

# Check sequence
def check_sequence(sequence, MIN_SEQUENCE_LENGTH = 16, MAX_SEQUENCE_LENGTH = 2500):
    # Remove all whitespaces, tabs and end lines; upper-case
    sequence = sequence.translate(str.maketrans('', '', ' \n\t')).upper()
    aatypes = set('ACDEFGHIKLMNPQRSTVWY')  # 20 standard aatypes
    if not set(sequence).issubset(aatypes):
        raise Exception(f'Input sequence contains non-amino acid letters: {set(sequence) - aatypes}. AlphaFold only supports 20 standard amino acids as inputs.')
    if len(sequence) < MIN_SEQUENCE_LENGTH:
        raise Exception(f'Input sequence is too short: {len(sequence)} amino acids, while the minimum is {MIN_SEQUENCE_LENGTH}')
    if len(sequence) > MAX_SEQUENCE_LENGTH:
        raise Exception(f'Input sequence is too long: {len(sequence)} amino acids, while the maximum is {MAX_SEQUENCE_LENGTH}. Please use the full AlphaFold system for long sequences.')
    return sequence

# Make table of sequence databases to be searched
def sequence_databases():
    mirror = fastest_sequence_db_mirror()
    db_prefix = f'https://storage.googleapis.com/alphafold-colab{mirror}/latest'
    databases = [
        {
            'name': 'uniref90',
            'url': db_prefix + '/uniref90_2021_03.fasta',
            'num chunks':59,
            'max hits': 10000,
            'z value': 135301051
        },
        {
            'name': 'smallbfd',
            'url': db_prefix + '/bfd-first_non_consensus_sequences.fasta',
            'num chunks': 17,
            'max hits': 10000,
            'z value': 65984053,
        },
        {
            'name': 'mgnify',
            'url': db_prefix + '/mgy_clusters_2019_05.fasta',
            'num chunks': 71,
            'max hits': 500,
            'z value': 304820129,
        },
    ]
    if fast_test:
        databases = [db for db in databases if db['name'] == 'smallbfd']
        databases[0]['num chunks'] = 5
    return databases

# Find the fastest responding mirror for sequence databases
def fastest_sequence_db_mirror(test_url_pattern = 'https://storage.googleapis.com/alphafold-colab{:s}/latest/uniref90_2021_03.fasta.1'):
    print ('Finding fastest mirror for sequence databases', end = '')
    from concurrent import futures
    ex = futures.ThreadPoolExecutor(3)
    def fetch(source):
        from urllib import request
        request.urlretrieve(test_url_pattern.format(source))
        return source
    fs = [ex.submit(fetch, source) for source in ['', '-europe', '-asia']]
    source = None
    for f in futures.as_completed(fs):
      source = f.result()
      ex.shutdown()
      break
    mirror = (source[1:] if source else 'united states')
    print (' using', mirror)
    return source

# Search against 1 Gbyte chunks of sequence databases streamed from the web.
def jackhmmer_sequence_search(seq_file, databases, jackhmmer_binary_path = '/usr/bin/jackhmmer'):

    dbs = []
    for db in databases:
        db_name = db['name']
        nchunks = db['num chunks']
        print ('Searching %s sequence database, %d Gbytes' % (db_name, nchunks))
        def progress_cb(i):
            print (' %d' % i, end = ('\n' if i%30 == 0 else ''), flush = True)

        from alphafold.data.tools import jackhmmer
        jackhmmer_runner = jackhmmer.Jackhmmer(
            binary_path=jackhmmer_binary_path,
            database_path=db['url'],
            get_tblout=True,
            num_streamed_chunks=db['num chunks'],
            streaming_callback = progress_cb,
            z_value=db['z value'])
        dbs.append((db_name, jackhmmer_runner.query(seq_file), db['max hits']))
        print ('')

    return dbs

# Extract the multiple sequence alignments from the Stockholm files.
def multiple_seq_align(dbs):
    msas = []
    deletion_matrices = []
    seen_already = set()
    db_counts = []
    for db_name, db_results, max_hits in dbs:
      unsorted_results = []
      for i, result in enumerate(db_results):
        from alphafold.data import parsers
        msa, deletion_matrix, target_names = parsers.parse_stockholm(result['sto'])
        e_values_dict = parsers.parse_e_values_from_tblout(result['tbl'])
        e_values = [e_values_dict[t.split('/')[0]] for t in target_names]
        zipped_results = zip(msa, deletion_matrix, target_names, e_values)
        if i != 0:
          # Only take query from the first chunk
          zipped_results = [x for x in zipped_results if x[2] != 'query']
        unsorted_results.extend(zipped_results)
      sorted_by_evalue = sorted(unsorted_results, key=lambda x: x[3])
      db_msas, db_deletion_matrices, _, _ = zip(*sorted_by_evalue)

      # Remove duplicates
      db_msas_uniq = []
      db_deletion_matrices_uniq = []
      for msa, dmat in zip(db_msas, db_deletion_matrices):
          if msa not in seen_already:
              seen_already.add(msa)
              db_msas_uniq.append(msa)
              db_deletion_matrices_uniq.append(dmat)
      db_msas, db_deletion_matrices = db_msas_uniq, db_deletion_matrices_uniq

      if db_msas:
        if max_hits is not None:
          db_msas = db_msas[:max_hits]
          db_deletion_matrices = db_deletion_matrices[:max_hits]
        msas.append(db_msas)
        deletion_matrices.append(db_deletion_matrices)
        db_counts.append((db_name, len(db_msas)))

    total = sum([count for name, count in db_counts], 0)
    counts = ', '.join('%d %s' % (count,name) for name, count in db_counts)
    print('%d similar sequences found (%s)' % (total, counts))
    return msas, deletion_matrices

# Predict the structures
def predict_structure(sequence, msas, deletion_matrices, model_names):
    plddts = {}
    pae_outputs = {}
    unrelaxed_proteins = {}

    num_templates = 0
    num_res = len(sequence)
    print('Computing structures using %d AlphaFold parameter sets:' % len(model_names))
    for model_name in model_names:
        print(' ' + model_name, end = '', flush = True)
        feature_dict = {}
        from alphafold.data import pipeline
        feature_dict.update(pipeline.make_sequence_features(sequence, 'test', num_res))
        feature_dict.update(pipeline.make_msa_features(msas, deletion_matrices=deletion_matrices))
        feature_dict.update(_placeholder_template_feats(num_templates, num_res))

        from alphafold.model import config, data, model
        cfg = config.model_config(model_name)
        params = data.get_model_haiku_params(model_name, './alphafold/data')
        model_runner = model.RunModel(cfg, params)
        processed_feature_dict = model_runner.process_features(feature_dict, random_seed=0)
        prediction_result = model_runner.predict(processed_feature_dict)

        if 'predicted_aligned_error' in prediction_result:
            pae_outputs[model_name] = (
                prediction_result['predicted_aligned_error'],
                prediction_result['max_predicted_aligned_error']
            )
        plddts[model_name] = prediction_result['plddt']

        # Set the b-factors to the per-residue plddt.
        final_atom_mask = prediction_result['structure_module']['final_atom_mask']
        b_factors = prediction_result['plddt'][:, None] * final_atom_mask
        from alphafold.common import protein
        unrelaxed_protein = protein.from_prediction(processed_feature_dict,
                                                    prediction_result,
                                                    b_factors=b_factors)
        unrelaxed_proteins[model_name] = unrelaxed_protein

        # Delete unused outputs to save memory.
        del model_runner
        del params
        del prediction_result
    print('')
    
    return unrelaxed_proteins, plddts, pae_outputs

def write_unrelaxed_pdbs(unrelaxed_proteins, pae_outputs, output_dir):
    # Write out PDB files and predicted alignment error
    from alphafold.common import protein
    for model_name, unrelaxed_protein in unrelaxed_proteins.items():
        write_pdb(protein.to_pdb(unrelaxed_protein), model_name + '_unrelaxed.pdb', output_dir)
        # Save predicted aligned error (if it exists)
        if model_name in pae_outputs:
            import os.path
            pae_output_path = os.path.join(output_dir, model_name + '_pae.json')
            save_predicted_aligned_error(pae_outputs[model_name], pae_output_path)

def write_best_pdb(plddts, unrelaxed_proteins, output_dir):
    # Find the best model according to the mean pLDDT.
    best_model_name = max(plddts.keys(), key=lambda x: plddts[x].mean())

    # AMBER relax the best model
    print('Energy minimizing best structure with OpenMM')
    relaxed_pdb = energy_minimize_structure(unrelaxed_proteins[best_model_name])

    # Write out the prediction
    write_pdb(relaxed_pdb, best_model_name + '_relaxed.pdb', output_dir)
    write_pdb(relaxed_pdb, 'best_model.pdb', output_dir)

def energy_minimize_structure(pdb_model):
    from alphafold.relax import relax
    amber_relaxer = relax.AmberRelaxation(
        max_iterations=0,
        tolerance=2.39,
        stiffness=10.0,
        exclude_residues=[],
        max_outer_iterations=20)
    relaxed_pdb, _, _ = amber_relaxer.process(prot=pdb_model)
    return relaxed_pdb

def write_pdb(pdb_model, filename, output_dir):
    import os.path
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
      f.write(pdb_model)

def _placeholder_template_feats(num_templates_, num_res_):
  from numpy import zeros, float32
  return {
      'template_aatype': zeros([num_templates_, num_res_, 22], float32),
      'template_all_atom_masks': zeros([num_templates_, num_res_, 37, 3], float32),
      'template_all_atom_positions': zeros([num_templates_, num_res_, 37], float32),
      'template_domain_names': zeros([num_templates_], float32),
      'template_sum_probs': zeros([num_templates_], float32),
  }

def save_predicted_aligned_error(model_pae, pae_output_path):
  # Save predicted aligned error in the same format as the AF EMBL DB
  pae, max_pae = model_pae
  import numpy as np
  rounded_errors = np.round(pae.astype(np.float64), decimals=1)
  indices = np.indices((len(rounded_errors), len(rounded_errors))) + 1
  indices_1 = indices[0].flatten().tolist()
  indices_2 = indices[1].flatten().tolist()
  pae_data = [{
      'residue1': indices_1,
      'residue2': indices_2,
      'distance': rounded_errors.flatten().tolist(),
      'max_predicted_aligned_error': max_pae.item()
  }]
  import json
  json_data = json.dumps(pae_data, indent=None, separators=(',', ':'))
  with open(pae_output_path, 'w') as f:
    f.write(json_data)

def predict_and_save(sequence, databases, output_dir = 'prediction',
                     model_names = ['model_1', 'model_2', 'model_3', 'model_4', 'model_5']):
    '''
    Model names refer to alphafold parameter sets.
    One structure is calculated for each model name.
    Model names ending in "_ptm" predict TM score ('model_1_ptm', ..., 'model_5_ptm').
    '''

    # Write target sequence to file in FASTA format for doing search.
    seq_file = 'target.fasta'
    with open(seq_file, 'wt') as f:
      f.write(f'>query\n{sequence}')

    # Search for sequences
    nchunks = sum(db['num chunks'] for db in databases)
    print ('Searching sequence databases (%d Gbytes).' % nchunks)
    print ('Search will take %d minutes or more.' % max(1,nchunks//5))
    dbs = jackhmmer_sequence_search(seq_file, databases)

    # Make multiple sequence alignment.
    print ('Computing multiple sequence alignment')
    msas, deletion_matrices = multiple_seq_align(dbs)

    # Create directory for writing structure files
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Predict structures
    unrelaxed_proteins, plddts, pae_outputs = \
        predict_structure(sequence, msas, deletion_matrices, model_names)

    # Write out PDB files and predicted errors
    write_unrelaxed_pdbs(unrelaxed_proteins, pae_outputs, output_dir)
    write_best_pdb(plddts, unrelaxed_proteins, output_dir)

    print ('Structure prediction completed.')
    
def set_environment_variables():
    # Set memory management environment variables used by AlphaFold dependencies
    import os
    os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '2.0'

def run_prediction(sequence, output_dir = 'prediction', INSTALL_LOG = 'install_log.txt'):
    '''Installs alphafold if not yet installed and runs a stucture prediction.'''

    # Check sequence length are within limits and no illegal characters
    sequence = check_sequence(sequence)
    print ('Sequence length %d' % len(sequence))
    
    # Check for GPU at beginning.
    # If no GPU then enabling a GPU runtime clears all virtual machine state
    # so need to enable GPU runtime before installing the prerequisites.
    check_for_gpu()

    if not is_alphafold_installed():
        print ('Installing HMMER for computing sequence alignments')
        install_hmmer(INSTALL_LOG = INSTALL_LOG)
        print ('Installing AlphaFold')
        install_alphafold(INSTALL_LOG = INSTALL_LOG)
        print ('Installing OpenMM for structure energy minimization')
        install_openmm(INSTALL_LOG = INSTALL_LOG)

    set_environment_variables()
    databases = sequence_databases()

    if fast_test:
        predict_and_save(sequence, databases, output_dir, model_names = ['model_1'])
    else:
        predict_and_save(sequence, databases, output_dir)

    # Make a zip file of the predictions
#    !zip -q -r {output_dir}.zip {output_dir}

    # Download predictions.  Does not work on Safari 14.1, macOS 10.15.7
    from google.colab import files
    files.download(f'{output_dir}/best_model.pdb')
#    files.download(f'{output_dir}.zip')

# ================================================================================================
# Predict a structure for a sequence.
#
fast_test = False
sequence = 'Paste a sequence here'  #@param {type:"string"}
#sequence = 'QVQLVESGGGSVQAGGSLRLSCTASGGSEYSYSTFSLGWFRQAPGQEREAVAAIASMGGLTYYADSVKGRFTISRDNAKNTVTLQMNNLKPEDTAIYYCAAVRGYFMRLPSSHNFRYWGQGTQVTVSSRGR'  #@param {type:"string"}

run_prediction(sequence)
