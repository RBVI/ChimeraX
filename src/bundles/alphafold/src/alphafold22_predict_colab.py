# ================================================================================================
# Google Colab code for running an AlphaFold structure prediction.
#

# Make sure Google Colab virtual machine has a GPU
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
        alphafold_version = 'v2.2.0',
        alphafold_parameters = ' https://storage.googleapis.com/alphafold/alphafold_params_2022-03-02.tar',
        bond_parameters = 'https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt',
        install_log = 'install_log.txt'):

    params_dir = './alphafold/data/params'
    import os.path
    params_path = os.path.join(params_dir, os.path.basename(alphafold_parameters))

    cmds = f'''
# Exit if any command fails
set -e

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
pip3 install --upgrade jaxlib==0.3.2+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip3 install jax==0.3.4
pip3 install --no-dependencies ./alphafold
#pip3 install ./alphafold

# Get AlphaFold parameters, 3.5 Gbytes,
# Ten models model_1, model_2, ..., model_5, model_1_ptm, ..., model_5_ptm.
mkdir -p "{params_dir}"
wget -q -O "{params_path}" {alphafold_parameters}
tar --extract --verbose --file="{params_path}" --directory="{params_dir}" --preserve-permissions
rm "{params_path}"

# Get standard bond length and bond angle parameters
mkdir -p /content/alphafold/alphafold/common
wget -q -P /content/alphafold/alphafold/common {bond_parameters}
cp -f /content/alphafold/alphafold/common/stereo_chemical_props.txt /usr/local/lib/python3.7/dist-packages/alphafold/common/

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
    if _exit_code != 0:
        raise RuntimeError('Error running shell script %s, output in log file %s'
                           % (filename, install_log))
    
def install_hmmer(install_log = 'install_log.txt'):
    # Install HMMER package in /usr/bin
    cmds = '''sudo apt install --quiet --yes hmmer'''
    run_shell_commands(cmds, 'install_hmmer.sh', install_log)

def install_matplotlib(install_log = 'install_log.txt'):
    # Install matplotlib for plotting alignment coverage
    cmds = '''pip install matplotlib'''
    run_shell_commands(cmds, 'install_matplotlib.sh', install_log)

def is_openmm_installed():
    try:
        import simtk.openmm
    except:
        return False
    return True
    
def install_openmm(
        conda_install_sh = 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh',
        install_log = 'install_log.txt'):
    '''Must install alphafold first since an openmm patch from alphafold is used.'''
    # Install Conda
    import os.path
    conda_install = os.path.join('/tmp', os.path.basename(conda_install_sh))
    cmds = f'''
# Exit if any command fails
set -e

wget -q -P /tmp {conda_install_sh} \
    && bash "{conda_install}" -b -p /opt/conda -f \
    && rm "{conda_install}"

# Install Python, OpenMM and pdbfixer in Conda
/opt/conda/bin/conda update -qy conda && \
    /opt/conda/bin/conda install -qy -c conda-forge python=3.7 openmm=7.5.1 cudatoolkit=11.2.2 pdbfixer

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
def check_sequences(sequences, min_sequence_length = 16, max_sequence_length = 2500):
    seqs = []
    for seq in sequences:
        # Remove all whitespaces, tabs and end lines; upper-case
        seq = seq.translate(str.maketrans('', '', ' \n\t')).upper()
        aatypes = set('ACDEFGHIKLMNPQRSTVWY')  # 20 standard aatypes
        if not set(seq).issubset(aatypes):
            raise Exception(f'Input sequence contains non-amino acid letters: {set(seq) - aatypes}. AlphaFold only supports 20 standard amino acids as inputs.')
        if len(seq) < min_sequence_length:
            raise Exception(f'Input sequence is too short: {len(seq)} amino acids, while the minimum is {min_sequence_length}')
        if len(seq) > max_sequence_length:
            raise Exception(f'Input sequence is too long: {len(seq)} amino acids, while the maximum is {max_sequence_length}. Please use the full AlphaFold system for long sequences.')
        seqs.append(seq)
    return seqs
    
def set_environment_variables():
    # Set memory management environment variables used by TensorFlow and JAX
    # These settings were suggested for longer sequences by SBGrid
    #  https://sbgrid.org/wiki/examples/alphafold2
    import os
    os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

# Create directory for results and write target sequence file.
def start_run(sequences, output_dir):
    # Move previous results if sequence has changed.
    from os import path, makedirs, rename
    seq_files = [path.join(output_dir, 'sequence_%d.fasta' % (i+1)) for i in range(len(sequences))]
    if path.exists(seq_files[0]):
        if same_sequences(sequences, seq_files):
            return seq_files
        # Rename current results directory and zip file.
        suffix = next_available_file_suffix(output_dir)
        rename(output_dir, output_dir + suffix)
        results_file = path.join(output_dir, '..', )
        if path.exists('results.zip'):
            rename('results.zip', 'results%s.zip' % suffix)

    # Make new results directory
    makedirs(output_dir, exist_ok=True)

    # Write target sequence to file in FASTA format for doing search.
    for seq, file in zip(sequences, seq_files):
        with open(file, 'wt') as f:
            f.write(f'>query\n{seq}')

    return seq_files

def same_sequences(sequences, seq_files):
    from os import path
    for seq, file in zip(sequences, seq_files):
        if not path.exists(file):
            return False
        last_seq = read_sequence(file)
        if seq != last_seq:
            return False
    return True
  
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
def sequence_databases(multimer = False):
    db_prefix = f'https://storage.googleapis.com/alphafold-colab%s/latest'
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
            'max hits': 5000,
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
    if multimer:
        databases.append({
            # Swiss-Prot and TrEMBL are concatenated together as UniProt.
            'name': 'uniprot',
            'url': db_prefix + '/uniprot_2021_03.fasta',
            'num chunks': 98,
            'max hits': 50000,
            'z value': 219174961 + 565254})

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

# Search against 1 Gbyte chunks of a sequence database streamed from the web.
def jackhmmer_sequence_search(seq_file, database, mirror = '',
                              jackhmmer_binary_path = '/usr/bin/jackhmmer'):

    db_name = database['name']
    nchunks = database['num chunks']
    db_url = database['url'] % mirror

    print ('Searching %s sequence database, %d Gbytes' % (db_name, nchunks))
    def progress_cb(i):
        print (' %d' % i, end = ('\n' if i%30 == 0 else ''), flush = True)

    from alphafold.data.tools import jackhmmer
    jackhmmer_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=db_url,
        get_tblout=True,
        num_streamed_chunks=database['num chunks'],
        streaming_callback = progress_cb,
        z_value=database['z value'])

    results = jackhmmer_runner.query(seq_file)

    print ('')

    return results

def write_sequence_alignment(msa, database_name, target, output_dir):
    prefix = '%s_%s' % (target, database_name)
    from os import path
    with open(path.join(output_dir, prefix + '_alignment'), 'w') as f:
        for line in msa.sequences:
            f.write(line + '\n')
    with open(path.join(output_dir, prefix + '_deletions'), 'w') as f:
        for dcounts in msa.deletion_matrix:
            f.write(','.join('%d' % count for count in dcounts) + '\n')

def read_sequence_alignment(database_name, target, output_dir):
    from os import path
    prefix = '%s_%s' % (target, database_name)
    apath = path.join(output_dir, prefix + '_alignment')
    dpath = path.join(output_dir, prefix + '_deletions')
    if not path.exists(apath) or not path.exists(dpath):
        return [],[]
    with open(apath, 'r') as f:
        seqs = [line.rstrip() for line in f.readlines()]
    with open(dpath, 'r') as f:
        dcounts = [[int(value) for value in line.split(',')] for line in f.readlines()]
    return seqs, dcounts

def search_sequence_databases(sequences, seq_files, databases, output_dir):
    seq_msas = []
    unique_seq_msas = {}
    for seq_index, (seq, file) in enumerate(zip(sequences, seq_files)):
        if seq in unique_seq_msas:
            import copy
            msas = copy.deepcopy(unique_seq_msas[seq])
        else:
            # Align
            msas = create_multiple_sequence_alignments(file, databases, output_dir)
            unique_seq_msas[seq] = msas
            plot = plot_alignment_coverage(msas)
            if plot:
                import os.path
                image_path = os.path.join(output_dir, 'sequence_coverage_%d.png' % (seq_index+1))
                plot.savefig(image_path, bbox_inches='tight')
        seq_msas.append(msas)
    return seq_msas

def create_multiple_sequence_alignments(sequence_file, databases, output_dir):
    '''Search and make one multiple sequence alignment for each database.'''

    from os.path import basename, splitext
    target = splitext(basename(sequence_file))[0]

    nchunks = sum(db['num chunks'] for db in databases)
    print ('Searching sequence databases (%d Gbytes).' % nchunks)
    print ('Search will take %d minutes or more.' % max(1,nchunks//5))

    # Search for sequences
    msas = []
    mirror = None
    from alphafold.notebooks import notebook_utils
    for database in databases:
        db_name = database['name']
        alignment, deletions = read_sequence_alignment(db_name, target, output_dir)
        if alignment:
            from alphafold.data.parsers import Msa
            descrips = ['%s %d' % (db_name,i+1) for i in range(len(alignment))]
            msa = Msa(alignment, deletions, descrips)
        else:
            if mirror is None:
                mirror = fastest_sequence_db_mirror()
            db_results = jackhmmer_sequence_search(sequence_file, database, mirror)
            # Make multiple sequence alignment.
            print ('Merging chunk sequence alignments for %s' % db_name)
            msa = notebook_utils.merge_chunked_msa(results=db_results, max_hits=database['max hits'])
            write_sequence_alignment(msa, db_name, target, output_dir)
        msas.append(msa)

    return msas

def plot_alignment_coverage(msas):
    counts = alignment_coverage([msa.sequences for msa in msas])
    if counts is None:
        return
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 3))
    plt.title('Number of Aligned Sequences with no Gap for each Residue Position')
    x = range(1, len(counts)+1)	# Start residue numbers at 1, not 0.
    plt.plot(x, counts, color='black')
    plt.xlabel('Residue number')
    plt.ylabel('Coverage')
    plot = plt.gcf()
    plt.show()
    return plot

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

# Create input for AlphaFold from sequence alignments.
def features(sequences, seq_msas):
    features_for_chain = {}
    heteromer = len(set(sequences)) > 1
    for seq_index, (seq, msas) in enumerate(zip(sequences, seq_msas)):
            
        # Create features dictionary
        feature_dict = {}
        from alphafold.data import pipeline
        feature_dict.update(pipeline.make_sequence_features(sequence=seq, description='query',
                                                            num_res=len(seq)))
        feature_dict.update(pipeline.make_msa_features(msas=msas))
        from alphafold.notebooks import notebook_utils
        placeholder_features = notebook_utils.empty_placeholder_template_features(num_templates=0,
                                                                                  num_res=len(seq))
        feature_dict.update(placeholder_features)

        # Construct the all_seq features only for heteromers, not homomers.
        if heteromer:
            from alphafold.data import msa_pairing
            valid_feats = msa_pairing.MSA_FEATURES + (
                'msa_uniprot_accession_identifiers',
                'msa_species_identifiers',
            )
            uniprot_msa = msas[-1]  # Last alignment is for uniprot database
            all_seq_features = {
                f'{k}_all_seq': v for k, v in pipeline.make_msa_features([uniprot_msa]).items()
                if k in valid_feats}
            feature_dict.update(all_seq_features)

        from alphafold.common import protein
        chain_id = protein.PDB_CHAIN_IDS[seq_index]
        features_for_chain[chain_id] = feature_dict

    # Do further feature post-processing depending on the model type.
    multimer = (len(sequences) > 1)
    if multimer:
        all_chain_features = {}
        from alphafold.data import pipeline_multimer
        for chain_id, chain_features in features_for_chain.items():
            all_chain_features[chain_id] = pipeline_multimer.convert_monomer_features(
                chain_features, chain_id)
        all_chain_features = pipeline_multimer.add_assembly_features(all_chain_features)
        from alphafold.data import feature_processing
        feature_dict = feature_processing.pair_and_merge(all_chain_features=all_chain_features)
    else:
        from alphafold.common import protein
        feature_dict = features_for_chain[protein.PDB_CHAIN_IDS[0]]

    return feature_dict

# Predict the structures
def predict_structure(feature_dict, multimer, model_name, output_dir):

    from alphafold.model import config, data, model
    cfg = config.model_config(model_name)
    if multimer:
        cfg.model.num_ensemble_eval = 1
    else:
        cfg.data.eval.num_ensemble = 1
    params = data.get_model_haiku_params(model_name, './alphafold/data')
    model_runner = model.RunModel(cfg, params)
    processed_feature_dict = model_runner.process_features(feature_dict, random_seed=0)
    prediction_result = model_runner.predict(processed_feature_dict, random_seed=0)

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
                                                b_factors=b_factors,
                                                remove_leading_feature_dimension = not multimer)

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

def best_model(model_names, output_dir):
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
        return None

    return best_model_name

def minimize_best_model(best_model_name, output_dir):
    # Energy minimize the best model
    from os import path
    relaxed_path = path.join(output_dir, best_model_name + '_relaxed.pdb')
    if path.exists(relaxed_path):
        return
    
    print('Energy minimizing best structure %s with OpenMM and Amber forcefield' % best_model_name)
    unrelaxed_path = path.join(output_dir, best_model_name + '_unrelaxed.pdb')
    from alphafold.common import protein
    with open(unrelaxed_path, 'r') as f:
        best_unrelaxed_protein = protein.from_pdb_string(f.read())
        relaxed_pdb = energy_minimize_structure(best_unrelaxed_protein)
        # Write out PDB file
        write_pdb(relaxed_pdb, best_model_name + '_relaxed.pdb', output_dir)

def energy_minimize_structure(pdb_model):
    from alphafold.relax import relax
    amber_relaxer = relax.AmberRelaxation(
        max_iterations=0,
        tolerance=2.39,
        stiffness=10.0,
        exclude_residues=[],
        max_outer_iterations=1,
        use_gpu=True)
    relaxed_pdb, _, _ = amber_relaxer.process(prot=pdb_model)
    return relaxed_pdb

def write_pdb(pdb_model, filename, output_dir):
    import os.path
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
      f.write(pdb_model)

def copy_file(filename, new_filename, output_dir):
    # Copy PAE file if it exists.
    from os import path
    from_path = path.join(output_dir, filename)
    if path.exists(from_path):
        to_path = path.join(output_dir, new_filename)
        import os
        if path.exists(to_path):
            os.remove(to_path)
        os.link(from_path, to_path)

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

def run_prediction(sequences,
                   model_names = None,
                   energy_minimize = True,
                   output_dir = 'prediction',
                   install_log = 'install_log.txt'):
    '''
    Installs alphafold if not yet installed and runs a stucture prediction.
    Model names ending in "_ptm" predict TM score ('model_1_ptm', ..., 'model_5_ptm').
    '''

    print('Using AlphaFold 2.2.0')
    
    # Check sequence length are within limits and no illegal characters
    sequences = check_sequences(sequences)
    if len(sequences) == 1:
        msg = 'Sequence length %d' % len(sequences[0])
    else:
        msg = ('%d sequences, total length %d (= %s)' %
               (len(sequences), sum(len(seq) for seq in sequences),
                ' + '.join('%d' % len(seq) for seq in sequences)))
    print(msg)

    # Check for GPU at beginning.
    # If no GPU then enabling a GPU runtime clears all virtual machine state
    # so need to enable GPU runtime before installing the prerequisites.
    check_for_gpu()

    # Install sequence search software, alphafold and OpenMM for energy minimization.
    if not is_alphafold_installed():
        print ('Installing HMMER for computing sequence alignments')
        install_hmmer(install_log = install_log)
        print ('Installing matplotlib to plot sequence alignment coverage')
        install_matplotlib(install_log = install_log)
        print ('Installing AlphaFold')
        install_alphafold(install_log = install_log)
    if energy_minimize and not is_openmm_installed():
        print ('Installing OpenMM for structure energy minimization')
        install_openmm(install_log = install_log)

    set_environment_variables()

    # Create directory for results and write sequence file.
    seq_files = start_run(sequences, output_dir)

    # Search sequence databases producing a multiple sequence alignments
    # for each sequence against each database.
    multimer = (len(sequences) > 1)
    databases = sequence_databases(multimer)
    seq_msas = search_sequence_databases(sequences, seq_files, databases, output_dir)

    # Create features dictionary input to AlphaFold
    feature_dict = features(sequences, seq_msas)

    # Choose which AlphaFold neural networks to use.
    if model_names is None:
        from alphafold.model import config
        model_names = config.MODEL_PRESETS['multimer' if multimer else 'monomer_ptm']
    if fast_test:
        model_names = model_names[:1]

    # Predict structures by running AlphaFold
    print('Computing structures using %d AlphaFold parameter sets:' % len(model_names))
    from os import path
    for model_name in model_names:
        if not path.exists(path.join(output_dir, model_name + '_unrelaxed.pdb')):
            print(' ' + model_name, end = '', flush = True)
            try:
                predict_structure(feature_dict, multimer, model_name, output_dir)
            except Exception:
                error_log_path = path.join(output_dir, model_name + '_error')
                import traceback
                with open(error_log_path, 'w') as f:
                    traceback.print_exc(file = f)
                print ('\nAlphaFold generated an error computing %s, error logged to %s\n'
                       % (model_name, error_log_path))
    print('')

    # Energy minimize
    best_model_name = best_model(model_names, output_dir)
    if best_model_name:
        if energy_minimize:
            minimize_best_model(best_model_name, output_dir)
        # Copy best model and pae files.
        pdb_suffix = '_relaxed.pdb' if energy_minimize else '_unrelaxed.pdb'
        copy_file(best_model_name + pdb_suffix, 'best_model.pdb', output_dir)
        copy_file(best_model_name + '_pae.json', 'best_model_pae.json', output_dir)
        
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
sequences = 'Paste a sequences separated by commas here'  #@param {type:"string"}

seq_list = sequences.split(',')
dont_minimize = (seq_list[0] == 'dont_minimize')
if dont_minimize:
    seq_list = seq_list[1:]

# Remove obsolete "prokaryote" flag
is_prokaryote = (seq_list[0] == 'prokaryote')
if is_prokaryote:
    seq_list = seq_list[1:]

run_prediction(seq_list, energy_minimize = not dont_minimize)
