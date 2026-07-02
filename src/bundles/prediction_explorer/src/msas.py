# Try to show OpenFold MSA in profile grid and color structure by conservation.

def show_openfold_msa(session, chain, color = False):
    alignment = _read_msa_alignment(chain, viewer = 'pg')
    if alignment is None:
        from chimerax.core.errors import UserError
        raise UserError(f'No MSA found for {chain}')

    # Highlight chain sequence on profile grid
    profile_grid = alignment.viewers[0]
    profile_grid.grid_canvas.choose_from_seq(chain)
    
    if color:
        cmd = f'color byattribute r:seq_identity {chain.atomspec} palette 50,white:100,red'
        from chimerax.core.commands import run
        run(session, cmd)

def _read_msa_alignment(chain, viewer = False):
    json_path, json_data, query_name = _openfold_json_input(chain.structure)
    chain_data = _openfold_chain_data(json_data, query_name, chain.chain_id)
    if chain_data is None:
        from chimerax.core.errors import UserError
        raise UserError('No specification for chain "{chain.chain_id}" in {json_path}')
    
    msa_npzs = chain_data.get('main_msa_file_paths')
    if msa_npzs is None:
        msa_npz = _colabfold_msa_paths(json_path, query_name, chain)
        if msa_npz:
            msa_npzs = [msa_npz]

    from os.path import isabs, join, dirname
    for i,msa_npz in enumerate(msa_npzs):
        if not isabs(msa_npz):
            # Make path absolute
            msa_npzs[i] = join(dirname(json_path), msa_npz)

    if msa_npzs is None:
        from chimerax.core.errors import UserError
        raise UserError('Could not find OpenFold MSA for "{chain.chain_id}" using {json_path}')
        
    sequences = []
    for path in msa_npzs:
        import numpy
        msa_data = numpy.load(path, allow_pickle = True)
        for k in msa_data.keys():
            arrays = msa_data[k].item()
            msa = arrays['msa']
            nseq = len(msa)
            #deletion_matrix = arrays['deletion_matrix']
            metadata = arrays['metadata']
            from chimerax.atomic import Sequence
            seqs = [Sequence(name = metadata[i].split(maxsplit=1)[0], characters = ''.join(msa[i]))
                    for i in range(nseq)]
            sequences.extend(seqs)

    if sequences:
        name = f'{query_name} MSA {len(seqs)}'
        session = chain.structure.session
        # Don't auto associate because it can make loading template structures very slow.
        alignment = session.alignments.new_alignment(seqs, name, auto_associate = False, viewer = viewer)
        alignment.associate(chain, seqs[0])  # For large alignments no auto-association is done.
        return alignment

    return None

def _openfold_chain_data(json_data, query_name, chain_id):
    queries = json_data['queries']
    for q_name, query_data in queries.items():
        if q_name == query_name:
            for component_info in query_data['chains']:
                if component_info['molecule_type'].lower() == 'protein':
                    chain_ids = component_info['chain_ids']
                    if chain_id in chain_ids:
                        return component_info
    return None

def _openfold_json_path(structure, return_query_name = False):
    cif_path = structure.filename # ~/Desktop/openfold/8rf4/8rf4/seed_42/8rf4_seed_42_sample_1_model.cif
    from os.path import dirname, basename, join
    seed_dir = dirname(cif_path)	# ~/Desktop/openfold/8rf4/8rf4/seed_42
    results_dir = dirname(seed_dir)	# ~/Desktop/openfold/8rf4/8rf4
    query_name = basename(results_dir)
    run_dir = dirname(results_dir)	# ~/Desktop/openfold/8rf4
#    json_path = join(run_dir, basename(run_dir) + '.json' )	# ~/Desktop/openfold/8rf4/8rf4.json
    json_path = join(run_dir, 'inference_query_set.json' )	# ~/Desktop/openfold/8rf4/8rf4.json
    if return_query_name:
        return json_path, query_name
    return json_path

def _openfold_json_input(structure):
    json_path, query_name = _openfold_json_path(structure, return_query_name = True)
    from os.path import exists
    if not exists(json_path):
        from chimerax.core.errors import UserError
        raise UserError(f'Did not find OpenFold input file {json_path}')

    import json
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    return json_path, json_data, query_name

def _is_openfold_model(structure):
    cif_path = structure.filename
    from os.path import basename
    cif_filename = basename(cif_path)
    return cif_filename.endswith('_model.cif') and '_seed_' in cif_filename and '_sample_' in cif_filename
    '''
    # Template structures are also 2 directories below inference_query_set.json.
    # So not a good test to look for that file.
    json_path = _openfold_json_path(structure)
    from os.path import exists
    return json_path and exists(json_path)
    '''
    
def _colabfold_msa_paths(json_path, query_name, chain):
    # MSA was not specified in input json.
    # If ChimeraX made OpenFold prediction using Colabfold server the
    # MSA will be in a colabfold_msas subdirectory.
    from os.path import join, dirname, exists
    msa_dir = join(dirname(json_path), 'colabfold_msas')
    if exists(msa_dir):
        mapping_json = join(msa_dir, 'mappings', 'chain_id_to_rep_id.json')
        if exists(mapping_json):
            with open(mapping_json, 'r') as f:
                import json
                mapping = json.load(f)
                key = f'{query_name}-{chain.chain_id}'
                msa_filename = mapping.get(key)
                if msa_filename:
                    msa_path = join(msa_dir, 'main', f'{msa_filename}.npz')
                    if exists(msa_path):
                        return msa_path
    return None

def _open_openfold_templates(chain, align = True, remove_nontemplate_chains = True, one_model_per_template = True):
    template_paths = _openfold_template_paths(chain)
    if len(template_paths) == 0:
        return [], []
    session = chain.structure.session
    from chimerax.core.commands import run, concise_model_spec
    template_models = []
    pdb_chain_ids = []
    for cif_path, (pdb_id, chain_id) in template_paths:
        pdb_chain_id = f'{pdb_id}_{chain_id}'
        pdb_chain_ids.append(pdb_chain_id)
        max_models = 'maxModels 1' if one_model_per_template else ''
        with session.in_script:  # Avoid saving templates in file history
            models = run(session, f'open {cif_path} {max_models} name {pdb_chain_id}')
        mspec = concise_model_spec(session, models)
        if remove_nontemplate_chains:
            run(session, f'delete {mspec} & ~/{chain_id}')
        if align:
            run(session, f'matchmaker {mspec} to {chain.atomspec} logParameters false')
        _show_ribbon_only(models)
        template_models.extend(models)
    # Put all template models under one parent model for easy hiding.
    tmspec = concise_model_spec(session, template_models)
    sname = chain.structure.name.split('_')[0]
    pname = f'{sname} {chain.chain_id} templates'
    if len(template_models) == 1:
        # Hack to work around rename not creating a grouping model if only one model specified.
        pname = pname[:-1] + ' ' + template_models[0].name
    pspec = template_models[0].atomspec
    run(session, f'rename {tmspec} "{pname}" id {pspec}')
    return template_models, pdb_chain_ids

def _show_ribbon_only(models):
    for m in models:
        m.atoms.displays = False
        m.residues.ribbon_displays = True

def _openfold_template_paths(chain):
    json_path, json_data, query_name = _openfold_json_input(chain.structure)
    chain_data = _openfold_chain_data(json_data, query_name, chain.chain_id)
    if chain_data is None:
        from chimerax.core.errors import UserError
        raise UserError(f'No input data for query {query_name} chain "{chain.chain_id}" in {json_path}')
    
    pdb_chains = chain_data.get('template_entry_chain_ids', [])
    template_alignments_path = chain_data.get('template_alignment_file_path')
    if template_alignments_path is None:
        return []
    from os.path import isabs, join, dirname
    if not isabs(template_alignments_path):
        # Make path absolute
        template_alignments_path = join(dirname(json_path), template_alignments_path)
    from os.path import dirname, join, exists
    align_dir = dirname(template_alignments_path)
    template_dir = dirname(align_dir)
    struct_dir = join(template_dir, 'template_structures')
    if not exists(struct_dir):
        return []
    template_paths = []
    for pdb_chain in pdb_chains:
        pdb_id, pdb_chain = pdb_chain.split('_')
        pdb_path = join(struct_dir, f'{pdb_id}.cif')
        if exists(pdb_path):
            template_paths.append((pdb_path, (pdb_id, pdb_chain)))
    return template_paths

def msa_taxonomy(session, chain, show_tree = False, min_sequences = 0, show_unbranched = True,
                 percent_identity_divisor = 'min',
                 show_uniprot_ids = False, subalignment = None):
    tax_tree, uniprot_ids, uniprot_seqs, sequences = _taxonomy_tree(chain,
                                                                    percent_identity_divisor=percent_identity_divisor)

    n_uniparc = len([seq for seq in sequences if seq.name.startswith('UniRef100_UPI')])
    header = f'Sequence names {len(uniprot_ids)} uniprot ids, {n_uniparc} uniparc, {len(sequences)} sequences'
    log = session.logger
    log.info(header)

    if show_uniprot_ids:
        log.info(', '.join(uniprot_ids))

    if show_tree:
        lines = tax_tree.taxonomy_tree_listing(count_threshold=min_sequences, show_unbranched = show_unbranched)
        
        tree_list = '\n'.join(lines)
        log.info(tree_list)

    if subalignment:
        nodes = tax_tree.all_nodes()
        align_nodes = [node for node in nodes
                       if node._taxonomy_name == subalignment
                       or node._taxonomy_common_name == subalignment]
        align_uids = []
        for node in align_nodes:
            align_uids.extend(node._uniprot_ids)
        id_to_seq = dict(zip(uniprot_ids, uniprot_seqs))
        aseqs = [id_to_seq[uid] for uid in align_uids]
        aseqs.insert(0, chain)
        alignment = _show_alignment(session, aseqs, name = subalignment)
        alignment.associate(chain, chain)
        
def _taxonomy_tree(chain, percent_identity_divisor = 'min'):
    alignment = _read_msa_alignment(chain)
    if alignment is None:
        from chimerax.core.errors import UserError
        raise UserError(f'No MSA found for {chain}')

    sequences = alignment.seqs
    uniprot_ids = []
    uniprot_seqs = []
    for seq in sequences:
        name = seq.name
        if name.startswith('UniRef100_') and not name.startswith('UniRef100_UPI'):
            uniprot_ids.append(name[10:])
            uniprot_seqs.append(seq)
    from .taxonomy import uniprot_ids_to_taxonomy_ids, uniprot_ids_to_taxonomy_tree
#    tax_ids = uniprot_ids_to_taxonomy_ids(uniprot_ids)
    tax_tree = uniprot_ids_to_taxonomy_tree(uniprot_ids)
    uid_to_identity = _compute_sequence_identities(sequences[0], uniprot_seqs, uniprot_ids,
                                                   percent_identity_divisor = percent_identity_divisor)
    tax_tree.assign_mean_value('mean_percent_identity', uid_to_identity)

    return tax_tree, uniprot_ids, uniprot_seqs, sequences

def _show_alignment(session, sequences, name):
    alignment = session.alignments.new_alignment(sequences, name, auto_associate = False)
    return alignment

def _compute_sequence_identities(sequence, uniprot_seqs, uniprot_ids, percent_identity_divisor = 'min'):
    return {uid:_sequence_identity(sequence, useq, percent_identity_divisor)
            for uid, useq in zip(uniprot_ids, uniprot_seqs)}

def _sequence_identity(seq1, seq2, percent_identity_divisor = 'min'):
    from numpy import frombuffer, uint8, count_nonzero
    c1, c2 = seq1.characters.encode('ascii'), seq2.characters.encode('ascii')	# Byte arrays
    gap = b'-'
    length = min(len(c1), len(c2.replace(gap,b''))) if percent_identity_divisor == 'min' else len(c1)
    return count_nonzero(frombuffer(c1, dtype=uint8) == frombuffer(c2, dtype=uint8)) / length

def register_msa_command(logger):
    from chimerax.core.commands import CmdDesc, register, BoolArg, IntArg, StringArg, EnumOf
    from chimerax.atomic import ChainArg
    desc = CmdDesc(
        required = [('chain', ChainArg)],
        keyword = [('color', BoolArg)],
        synopsis = 'Open OpenFold MSA as profile grid'
    )
    register('msa', desc, show_openfold_msa, logger=logger)

    desc = CmdDesc(
        required = [('chain', ChainArg)],
        keyword = [('show_tree', BoolArg),
                   ('min_sequences', IntArg),
                   ('show_unbranched', BoolArg),
                   ('percent_identity_divisor', EnumOf(['min', 'ref'])),
                   ('show_uniprot_ids', BoolArg),
                   ('subalignment', StringArg)],
        synopsis = 'Show UniProt taxonomy tree for an OpenFold MSA'
    )
    register('msa taxonomy', desc, msa_taxonomy, logger=logger)
