# vim: set expandtab shiftwidth=4 softtabstop=4:

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

results_directory = '~/Downloads/ChimeraX/ESMFold'

def esmfold_predict(session, sequence, residue_range=None, subsequence=None,
                    chunk=None, overlap=0):
    if not _is_esmfold_available(session):
        return

    if not hasattr(session, '_cite_esmfold'):
        msg = 'Please cite <a href="https://doi.org/10.1101/2022.07.20.500902">Language models of protein sequences at the scale of evolution enable accurate structure prediction.</a> if you use these predictions.'
        session.logger.info(msg, is_html = True)
        session._cite_esmfold = True  # Only log this message once per session.
        
    if _running_esmfold_prediction(session):
        from chimerax.core.errors import UserError
        raise UserError('ESMFold prediction currently running.  Can only run one at a time.')

    from chimerax.atomic import Chain
    chain = sequence if isinstance(sequence, Chain) else None
    seq = sequence.ungapped()

    if residue_range is not None:
        rbase = 1 if chain is None else chain.numbering_start
        r0,r1 = residue_range
        seq = seq[r0-rbase:r1-rbase+1]
            
    if subsequence is not None:
        first,last = subsequence
        seq = seq[first-1:last]

    if chunk is not None and len(seq) > chunk:
        if overlap >= chunk:
            from chimerax.core.errors import UserError
            raise UserError('esmfold predict requires overlap < chunk')
        offset = 0
        while True:
            cseq = seq[offset:offset+chunk]
            _start_esmfold_prediction(session, cseq, align_to = chain)
            if offset+chunk >= len(seq):
                break
            offset += chunk - overlap
    else:
        _start_esmfold_prediction(session, seq, align_to = chain)

# ------------------------------------------------------------------------------
#
def _is_esmfold_available(session):
    # TODO: Check status web page
    return True

# ------------------------------------------------------------------------------
#
def _running_esmfold_prediction(session):
    return False

# ------------------------------------------------------------------------------
#
esmfold_predict_url = 'https://api.esmatlas.com/foldSequence/v1/pdb'
def _start_esmfold_prediction(session, sequence, align_to = None):
    '''sequence is a Sequence or Chain instance.'''
    import requests
    r = requests.post(esmfold_predict_url, data = sequence)
    pdb_path = _esmfold_pdb_file_path(sequence)
    pdb_string = r.text

    if not pdb_string.startswith('HEADER'):
        from chimerax.core.errors import UserError
        raise UserError('Prediction failed server https://api.esmatlas.com limits sequence length and run time: ' + pdb_string)
    
    with open(pdb_path, 'w') as f:
        f.write(pdb_string)
    from chimerax.core.commands import run
    s = run(session, f'open {pdb_path}')[0]
    run(session, f'color bfactor {s.atomspec} palette esmfold')

    # Align to specified structure.
    if align_to is not None:
        chain = align_to
        from chimerax.alphafold.match import _rename_chains, _align_to_chain
        _rename_chains(s, [chain.chain_id])
        _align_to_chain(s, chain)

# ------------------------------------------------------------------------------
#
def _esmfold_pdb_file_path(sequence, seq_chars=10):
    from os import path, makedirs
    dir = path.expanduser(results_directory)
    if not path.exists(dir):
        makedirs(dir)
    basename = f'{sequence[:seq_chars]}_{len(sequence)}'
    basepath = path.join(dir, basename)
    pdb_path = _unique_filename(basepath, '.pdb')
    return pdb_path

# ------------------------------------------------------------------------------
#
def _unique_filename(basepath, suffix):
    path = basepath + suffix
    from os.path import exists
    if not exists(path):
        return path
    i = 2
    while True:
        path = f'{basepath}_{i}{suffix}'
        if not exists(path):
            break
        i += 1
    return path
    
# ------------------------------------------------------------------------------
#
def register_esmfold_predict_command(logger):
    from chimerax.core.commands import CmdDesc, register, Int2Arg, IntArg
    from chimerax.atomic import SequenceArg
    desc = CmdDesc(
        required = [('sequence', SequenceArg)],
        keyword = [('subsequence', Int2Arg),
                   ('residue_range', Int2Arg),
                   ('chunk', IntArg),
                   ('overlap', IntArg)],
        synopsis = 'Predict a structure with ESMFold'
    )
    register('esmfold predict', desc, esmfold_predict, logger=logger)

