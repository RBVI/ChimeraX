def open_mutation_scores_csv(session, path, chain = None, name = None):
    mset = _read_mutation_scores_csv(path)

    from .ms_data import mutation_scores_manager
    msm = mutation_scores_manager(session)
    if name is None:
        from os.path import basename, splitext
        name = splitext(basename(path))[0]
    mset.name = name
    msm.add_scores(name, mset)

    if chain:
        mset.chain = chain

    nmut = len(mset.mutation_scores)
    dresnums = set(mset.residue_number_to_amino_acid().keys())
    score_names = ', '.join(mset.score_names())
    message = f'Opened deep mutational scan data for {nmut} mutations of {len(dresnums)} residues with score names {score_names}.'
    
    if chain:
        cres = chain.existing_residues
        sresnums = set(r.number for r in cres)
        message += f' Assigned scores to {len(sresnums & dresnums)} of {len(cres)} residues of chain {chain}.'
        mres = len(dresnums - sresnums)
        if mres > 0:
            message += f' Found scores for {mres} residues not present in atomic model.'

    return mset, message

def _read_mutation_scores_csv(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    headings = [h.strip() for h in lines[0].split(',')]
    mscores = []
    mut = set()
    from .ms_data import MutationScores    
    for i, line in enumerate(lines[1:]):
        if line.strip() == '':
            continue	# Ignore blank lines
        fields = line.split(',')
        if len(fields) != len(headings):
            from chimerax.core.errors import UserError
            raise UserError(f'Line {i+2} has wrong number of comma-separated fields, got {len(fields)}, but there are {len(headings)} headings')
        hgvs = fields[0]
        if not hgvs.startswith('p.(') or not hgvs.endswith(')'):
            from chimerax.core.errors import UserError
            raise UserError(f'Line {i+2} has hgvs field "{hgvs}" not starting with "p.(" and ending with ")"')
        if 'del' in hgvs or 'ins' in hgvs or '_' in hgvs:
            continue
        res_type = hgvs[3]
        res_num = int(hgvs[4:-2])
        res_type2 = hgvs[-2]
        if (res_num, res_type, res_type2) in mut:
            from chimerax.core.errors import UserError
            raise UserError(f'Duplicated mutation "{hgvs}" at line {i+2}')
        mut.add((res_num, res_type, res_type2))
        scores = _parse_scores(headings, fields)
        mscores.append(MutationScores(res_num, res_type, res_type2, scores))

    from os.path import basename, splitext
    name = splitext(basename(path))[0]
    from .ms_data import MutationSet
    mset = MutationSet(name, mscores, path = path)

    return mset

def _parse_scores(headings, fields):
    scores = {}
    for h,f in zip(headings[1:], fields[1:]):
        try:
            scores[h] = float(f)
        except ValueError:
            continue
    return scores
