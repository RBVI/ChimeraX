# ================================================================================================
# Google Colab code for running an AlphaFold structure prediction.
#

# Make sure virtual machine has a GPU
def check_for_gpu():
    import os
    have_gpu = (int(os.environ.get('COLAB_GPU',1)) > 0)
    if have_gpu:
        print ('Have Colab GPU runtime')
    else:
        raise RuntimeError('Require Colab GPU runtime.\n' +
                           'Change GPU with Colab menu\n' +
                           'Runtime -> Change Runtime Type -> Hardware accelerator -> GPU.')

def is_alphafold_installed():
    try:
        import alphafold
    except:
        return False
    return True

def install_alphafold(
        alphafold_git_repo = 'https://github.com/deepmind/alphafold',
        alphafold_version = 'v2.0.1',
        alphafold_parameters = 'https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar',
        bond_parameters = 'https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt',
        install_log = 'install_log.txt'):

    params_dir = './alphafold/data/params'
    import os.path
    params_path = os.path.join(params_dir, os.path.basename(alphafold_parameters))

    cmds = f'''
# Uninstall Google Colab default tensorflow
pip3 uninstall -y tensorflow

# Get AlphaFold from GitHub and install it
git clone --branch {alphafold_version} {alphafold_git_repo} alphafold
# Install versions of dependencies specified in requirements.txt
# Alphafold fails because jax==0.2.14 is incompatible with much newer jaxlib=0.1.70
# resulting in error no module jax.experimental.compilation_cache.  The chex
# package brings in jax 0.2.19 and jaxlib 0.1.70 but then jax is uninstalled
# and replaced with 0.2.14 but jaxlib is not reverted to an older version.
# Also need to get jaxlib from google rather than pypi to have cuda support.
pip3 install -r ./alphafold/requirements.txt
# Update jax
pip3 install --upgrade jaxlib==0.1.70+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip3 install jax==0.2.19
pip3 install --no-dependencies ./alphafold
#pip3 install ./alphafold

# Get AlphaFold parameters, 3.5 Gbytes,
# Ten models model_1, model_2, ..., model_5, model_1_ptm, ..., model_5_ptm.
mkdir -p "{params_dir}"
wget -q -O "{params_path}" {alphafold_parameters}
tar --extract --verbose --file="{params_path}" --directory="{params_dir}" --preserve-permissions
rm "{params_path}"

# Get standard bond length and bond angle parameters
mkdir -p /content/alphafold/common
wget -q -P /content/alphafold/common {bond_parameters}

# Create a ramdisk to store a database chunk to make jackhmmer run fast.
# Module alphafold.data.tools.jackhmmer makes use of this /tmp/ramdisk.
sudo mkdir -m 777 --parents /tmp/ramdisk
sudo mount -t tmpfs -o size=9G ramdisk /tmp/ramdisk
'''
    run_shell_commands(cmds, 'install_alphafold.sh', install_log)
    
def run_shell_commands(commands, filename, install_log):
    with open(filename, 'w') as f:
        f.write(commands)

    # The -x option logs each command with a prompt in front of it.
    !bash -x "{filename}" >> "{install_log}" 2>&1
    
def install_hmmer(install_log = 'install_log.txt'):
    # Install HMMER package in /usr/bin
    cmds = '''sudo apt install --quiet --yes hmmer'''
    run_shell_commands(cmds, 'install_hmmer.sh', install_log)

def install_matplotlib(install_log = 'install_log.txt'):
    # Install matplotlib for plotting alignment coverage
    cmds = '''pip install matplotlib'''
    run_shell_commands(cmds, 'install_matplotlib.sh', install_log)
    
def install_openmm(
        conda_install_sh = 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh',
        install_log = 'install_log.txt'):
    '''Must install alphafold first since an openmm patch from alphafold is used.'''
    # Install Conda
    import os.path
    conda_install = os.path.join('/tmp', os.path.basename(conda_install_sh))
    cmds = f'''
wget -q -P /tmp {conda_install_sh} \
    && bash "{conda_install}" -b -p /opt/conda \
    && rm "{conda_install}"

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
    run_shell_commands(cmds, 'install_openmm.sh', install_log)

# ================================================================================================
# Python routines to run a prediction.
#

# Check sequence
def check_sequence(sequence, min_sequence_length = 16, max_sequence_length = 2500):
    # Remove all whitespaces, tabs and end lines; upper-case
    sequence = sequence.translate(str.maketrans('', '', ' \n\t')).upper()
    aatypes = set('ACDEFGHIKLMNPQRSTVWY')  # 20 standard aatypes
    if not set(sequence).issubset(aatypes):
        raise Exception(f'Input sequence contains non-amino acid letters: {set(sequence) - aatypes}. AlphaFold only supports 20 standard amino acids as inputs.')
    if len(sequence) < min_sequence_length:
        raise Exception(f'Input sequence is too short: {len(sequence)} amino acids, while the minimum is {min_sequence_length}')
    if len(sequence) > max_sequence_length:
        raise Exception(f'Input sequence is too long: {len(sequence)} amino acids, while the maximum is {max_sequence_length}. Please use the full AlphaFold system for long sequences.')
    return sequence

# Create directory for results and write target sequence file.
def start_run(sequence, output_dir):
    # Move previous results if sequence has changed.
    from os import path, makedirs, rename
    seq_file = path.join(output_dir, 'target.fasta')
    if path.exists(seq_file):
        last_seq = read_sequence(seq_file)
        if sequence == last_seq:
            return seq_file
        # Rename current results directory and zip file.
        suffix = next_available_file_suffix(output_dir)
        rename(output_dir, output_dir + suffix)
        results_file = path.join(output_dir, '..', )
        if path.exists('results.zip'):
            rename('results.zip', 'results%s.zip' % suffix)

    # Make new results directory
    makedirs(output_dir, exist_ok=True)

    # Write target sequence to file in FASTA format for doing search.
    seq_file = path.join(output_dir, 'target.fasta')
    with open(seq_file, 'wt') as f:
      f.write(f'>query\n{sequence}')

    return seq_file
  
def read_sequence(seq_file):
    with open(seq_file, 'r') as f:
        return ''.join(line.strip() for line in f.readlines()[1:])

def next_available_file_suffix(path):
    i = 1
    import os.path
    while os.path.exists(path + ('%d' % i)):
        i += 1
    return '%d' % i
    
# Make table of sequence databases to be searched
def sequence_databases():
    db_prefix = f'https://storage.googleapis.com/alphafold-colab%s/latest'
    databases = [
        {
            'name': 'uniref90',
            'url': db_prefix + '/uniref90_2021_03.fasta',
            'num chunks':59,
            'max hits': None,		# Tried 10000 to avoid out of memory
            'z value': 135301051
        },
        {
            'name': 'smallbfd',
            'url': db_prefix + '/bfd-first_non_consensus_sequences.fasta',
            'num chunks': 17,
            'max hits': None,		# Tried 10000 to avoid out of memory
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

def write_sequence_alignments(msas, deletion_matrices, db_names, output_dir):
    from os import path
    for msa, deletions, name in zip(msas, deletion_matrices, db_names):
        with open(path.join(output_dir, name + '_alignment'), 'w') as f:
            for line in msa:
                f.write(line + '\n')
        with open(path.join(output_dir, name + '_deletions'), 'w') as f:
            for dcounts in deletions:
                f.write(','.join('%d' % count for count in dcounts) + '\n')

def read_alignments(database_names, output_dir):
    alignments, deletions = [], []
    from os import path
    for name in database_names:
        apath = path.join(output_dir, name + '_alignment')
        dpath = path.join(output_dir, name + '_deletions')
        if not path.exists(apath) or not path.exists(dpath):
            return [],[]
        with open(apath, 'r') as f:
            seqs = [line.rstrip() for line in f.readlines()]
            alignments.append(seqs)
        with open(dpath, 'r') as f:
            dcounts = [[int(value) for value in line.split(',')] for line in f.readlines()]
            deletions.append(dcounts)
    return alignments, deletions

def create_multiple_sequence_alignment(sequence_file, databases, output_dir):
    db_names = [db['name'] for db in databases]
    alignments, deletions = read_alignments(db_names, output_dir)
    if alignments:
        return alignments, deletions

    # Find fastest database mirror
    mirror = fastest_sequence_db_mirror()
    for db in databases:
        db['url'] = db['url'] % mirror

    # Search for sequences
    nchunks = sum(db['num chunks'] for db in databases)
    print ('Searching sequence databases (%d Gbytes).' % nchunks)
    print ('Search will take %d minutes or more.' % max(1,nchunks//5))
    dbs = jackhmmer_sequence_search(sequence_file, databases)

    # Make multiple sequence alignment.
    print ('Computing multiple sequence alignment')
    alignments, deletions = multiple_seq_align(dbs)
    write_sequence_alignments(alignments, deletions, db_names, output_dir)

    return alignments, deletions

def plot_alignment_coverage(alignments):
    counts = alignment_coverage(alignments)
    if counts is None:
        return
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 3))
    plt.title('Number of Aligned Sequences with no Gap for each Residue Position')
    x = range(1, len(counts)+1)	# Start residue numbers at 1, not 0.
    plt.plot(x, counts, color='black')
    plt.xlabel('Residue number')
    plt.ylabel('Coverage')
    plt.show()

def alignment_coverage(alignments):
    counts = None
    for alignment in alignments:
        for line in alignment:
            if counts is None:
                from numpy import zeros, int32
                counts = zeros((len(line),), int32)
            for i,c in enumerate(line):
                if c != '-':
                    counts[i] += 1
    return counts

# Predict the structures
def predict_structure(sequence, msas, deletion_matrices, model_name, output_dir):
    num_templates = 0
    num_res = len(sequence)

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
        pae_output = (
            prediction_result['predicted_aligned_error'],
            prediction_result['max_predicted_aligned_error']
        )
    else:
        pae_output = None
    plddt = prediction_result['plddt']

    # Set the b-factors to the per-residue plddt.
    final_atom_mask = prediction_result['structure_module']['final_atom_mask']
    b_factors = prediction_result['plddt'][:, None] * final_atom_mask
    from alphafold.common import protein
    unrelaxed_protein = protein.from_prediction(processed_feature_dict,
                                                prediction_result,
                                                b_factors=b_factors)

    # Delete unused outputs to save memory.
    del model_runner
    del params
    del prediction_result

    score = plddt.mean()
    write_unrelaxed_pdb(model_name, unrelaxed_protein, score, pae_output, output_dir)
    
    return unrelaxed_protein, plddt, pae_output

def write_unrelaxed_pdb(model_name, unrelaxed_protein, score, pae_output, output_dir):
    # Write out PDB files and predicted alignment error
    from alphafold.common import protein
    write_pdb(protein.to_pdb(unrelaxed_protein), model_name + '_unrelaxed.pdb', output_dir)

    from os import path
    with open(path.join(output_dir, model_name + '_score'), 'w') as f:
        f.write('%.5g' % score)

    # Save predicted aligned error (if it exists)
    if pae_output is not None:
        pae_output_path = path.join(output_dir, model_name + '_pae.json')
        save_predicted_aligned_error(pae_output, pae_output_path)

def minimize_best_model(model_names, output_dir):
    from os import path
    if path.exists(path.join(output_dir, 'best_model.pdb')):
        return  # Already minimized

    best_score = None
    from os import path
    for name in model_names:
        spath = path.join(output_dir, name + '_score')
        if path.exists(spath):
            with open(spath, 'r') as f:
                score = float(f.readline())
            if best_score is None or score > best_score:
                best_score, best_model_name = score, name

    if best_score is None:
        print('No models successfully computed.')
        return
        
    # Energy minimize the best model
    print('Energy minimizing best structure %s with OpenMM and Amber forcefield' % best_model_name)
    from alphafold.common import protein
    with open(path.join(output_dir, best_model_name + '_unrelaxed.pdb'), 'r') as f:
        best_unrelaxed_protein = protein.from_pdb_string(f.read())
        relaxed_pdb = energy_minimize_structure(best_unrelaxed_protein)

    # Write out PDB file
    write_pdb(relaxed_pdb, best_model_name + '_relaxed.pdb', output_dir)
    write_pdb(relaxed_pdb, 'best_model.pdb', output_dir)

def energy_minimize_structure(pdb_model):
    from alphafold.relax import relax
    amber_relaxer = relax.AmberRelaxation(
        max_iterations=0,
        tolerance=2.39,
        stiffness=10.0,
        exclude_residues=[],
        max_outer_iterations=1)
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
    
def set_environment_variables():
    # Set memory management environment variables used by TensorFlow and JAX
    # These settings were suggested for longer sequences by SBGrid
    #  https://sbgrid.org/wiki/examples/alphafold2
    import os
    os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    
def run_prediction(sequence,
                   model_names = ['model_1', 'model_2', 'model_3', 'model_4', 'model_5'],
                   output_dir = 'prediction',
                   install_log = 'install_log.txt'):
    '''
    Installs alphafold if not yet installed and runs a stucture prediction.
    Model names ending in "_ptm" predict TM score ('model_1_ptm', ..., 'model_5_ptm').
    '''
    # Check sequence length are within limits and no illegal characters
    sequence = check_sequence(sequence)
    print ('Sequence length %d' % len(sequence))
    
    # Check for GPU at beginning.
    # If no GPU then enabling a GPU runtime clears all virtual machine state
    # so need to enable GPU runtime before installing the prerequisites.
    check_for_gpu()

    if not is_alphafold_installed():
        print ('Installing HMMER for computing sequence alignments')
        install_hmmer(install_log = install_log)
        print ('Installing matplotlib to plot sequence alignment coverage')
        install_matplotlib(install_log = install_log)
        print ('Installing AlphaFold')
        install_alphafold(install_log = install_log)
        print ('Installing OpenMM for structure energy minimization')
        install_openmm(install_log = install_log)

    set_environment_variables()
    databases = sequence_databases()

    if fast_test:
        model_names = model_names[:1]

    # Create directory for results and write sequence file.
    seq_file = start_run(sequence, output_dir)

    # Align
    alignments, deletions = create_multiple_sequence_alignment(seq_file, databases, output_dir)
    plot_alignment_coverage(alignments)

    # Predict
    print('Computing structures using %d AlphaFold parameter sets:' % len(model_names))
    from os import path
    for model_name in model_names:
        if not path.exists(path.join(output_dir, model_name + '_unrelaxed.pdb')):
            print(' ' + model_name, end = '', flush = True)
            try:
                predict_structure(sequence, alignments, deletions, model_name, output_dir)
            except Exception:
                error_log_path = path.join(output_dir, model_name + '_error')
                import traceback
                with open(error_log_path, 'w') as f:
                    traceback.print_exc(file = f)
                print ('\nAlphaFold generated an error computing %s, error logged to %s\n'
                       % (model_name, error_log_path))
    print('')

    # Energy minimize
    minimize_best_model(model_names, output_dir)
    print ('Structure prediction completed.')

    # Make a zip file of the predictions
    !cd {output_dir} ; zip -q -r ../results.zip *
    
    # Download predictions.
    from google.colab import files
    files.download('results.zip')

# ================================================================================================
# Predict a structure for a sequence.
#
fast_test = False
sequence = 'Paste a sequence here'  #@param {type:"string"}

run_prediction(sequence)
