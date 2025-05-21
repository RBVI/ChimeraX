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
#
def alphafold_dimers(session, sequences, with_sequences = None,
                     homodimers = True, max_length = None, recycles = 3,
                     models = (1,2,3,4,5), output_fasta = None, output_json = None):
    '''
    Create dimer sequence file to run predictions of all combinations for a given set of sequences
    using localcolabfold (https://github.com/YoshitakaMo/localcolabfold).
    '''
    from chimerax.core.errors import UserError
    if len(sequences) == 0:
        raise UserError('No sequences specified')
    if with_sequences is not None and len(with_sequences) == 0:
        raise UserError('No with sequences specified')

    seqs = _unique_sequences(sequences)
    with_seqs = _unique_sequences(with_sequences) if with_sequences is not None else None

    all_seq_pairs = _sequence_pairs(seqs, with_seqs, homodimers)
    seq_pairs, long_pairs = _filter_by_length(all_seq_pairs, max_length)

    msg = _prediction_info(seqs, with_seqs, seq_pairs, long_pairs, recycles, models)
    session.logger.info(msg)

    if output_fasta:
        write_dimers_fasta(output_fasta, seq_pairs)
        cmd = colabfold_batch_command(output_fasta, recycles, models)
        session.logger.info(f'Prediction command: {cmd}')

    if output_json:
        write_dimers_json(output_json, seq_pairs)
        
# -----------------------------------------------------------------------------
#
def run_predictions(fasta_path, recycles, models, directory):
    '''
    To start a subprocess which will continue to run even if ChimeraX exits use
       p = subprocess.Popen(cmd, start_new_session=True)
    to control which GPU is used
       env = {'CUDA_VISIBLE_DEVICES':'1'}
       p = subprocess.Popen(cmd, start_new_session=True, env=env)
    '''
    from subprocess import Popen

# -----------------------------------------------------------------------------
#
def colabfold_batch_command(fasta_path, recycles, models):
    options = []
    # colabfold_batch uses 20 recycles with early termination by default for multimers
    # but 3 recycles for monomers.  So always set the recycles option.
    options.append(f'--num-recycle {recycles}')
    if models != (1,2,3,4,5):
        modnums = ','.join(str(m) for m in models)
        options.append(f'--model-order {modnums}')

    from chimerax.core.commands import quote_path_if_necessary
    quoted_fasta_path = quote_path_if_necessary(fasta_path)
    output_directory = '.'
    opts = ' '.join(options)
    cmd = f'colabfold_batch {opts} {quoted_fasta_path} {output_directory}'
    return cmd

# -----------------------------------------------------------------------------
#
def write_dimers_fasta(output_fasta, seq_pairs):
    lines = []
    for (name1, seq1), (name2, seq2) in seq_pairs:
        lines.extend([f'>{name1}.{name2}',
                      seq1 + ':',
                      seq2])
    with open(output_fasta, 'w') as f:
        f.write('\n'.join(lines))

# -----------------------------------------------------------------------------
#
def write_dimers_json(output_json, seq_pairs, seeds = [1]):
    '''Write AlphaFold 3 server json format.'''
    jobs = []
    for (name1, seq1), (name2, seq2) in seq_pairs:
        job = {
            'name': f'{name1}-{name2}',
            'sequences': [{'proteinChain': {'sequence':seq1, 'count':1}},
                          {'proteinChain': {'sequence':seq2, 'count':1}}]
        }
        if seeds is not None:
            job['modelSeeds'] = [str(seed) for seed in seeds]
        jobs.append(job)
    import json
    with open(output_json, 'w') as f:
        json.dump(jobs, f)

# -----------------------------------------------------------------------------
#
def write_monomers_fasta(output_fasta, named_seqs):
    lines = []
    for name, seq in named_seqs:
        lines.extend([f'>{name}', seq])
    with open(output_fasta, 'w') as f:
        f.write('\n'.join(lines))

# -----------------------------------------------------------------------------
#
def write_monomers_json(output_json, named_seqs, seeds = [1]):
    '''Write AlphaFold 3 server json format.'''
    jobs = []
    for name, seq in named_seqs:
        job = {
            'name': name,
            'sequences': [{'proteinChain': {'sequence':seq, 'count':1}}]
        }
        if seeds is not None:
            job['modelSeeds'] = [str(seed) for seed in seeds]
        jobs.append(job)
    import json
    with open(output_json, 'w') as f:
        json.dump(jobs, f)
        
# -----------------------------------------------------------------------------
#
def _unique_sequences(named_sequences):
    useqs = []
    nseqs = set()
    names = set()
    seqs = set()
    for name, seq in named_sequences:
        if (name,seq) in nseqs:
            continue
        elif name in names:
            from chimerax.core.errors import UserError
            raise UserError('Found two different sequences with the same name "{name}"')
        elif seq in seqs:
            continue	# Repeat
        useqs.append((name, seq))
        nseqs.add((name,seq))
        names.add(name)
        seqs.add(seq)
    return useqs

# -----------------------------------------------------------------------------
#
def _sequence_pairs(seqs, with_seqs = None, homodimers = True):
    seq_pairs = []
    seqs2 = seqs if with_seqs is None else with_seqs
    found = set()
    for name1,seq1 in seqs:
        for name2,seq2 in seqs2:
            if (seq1, seq2) not in found and (seq2, seq1) not in found:
                if homodimers or seq2 != seq1:
                    seq_pairs.append(((name1,seq1),(name2,seq2)))
                    found.add((seq1, seq2))
    return seq_pairs

# -----------------------------------------------------------------------------
#
def _filter_by_length(all_seq_pairs, max_length):
    if max_length is None:
        return all_seq_pairs, []
    seq_pairs = []
    long_pairs = []
    for p in all_seq_pairs:
        (name1,seq1),(name2,seq2) = p
        if len(seq1)+len(seq2) <= max_length:
            seq_pairs.append(p)
        else:
            long_pairs.append(p)
    return seq_pairs, long_pairs

# -----------------------------------------------------------------------------
#
def _sequence_pair_lengths(seq_pairs):
    return [len(seq1)+len(seq2) for (name1,seq1),(name2,seq2) in seq_pairs]

# -----------------------------------------------------------------------------
#
def _estimated_runtime(sequence_length, recycles = 3, num_models = 5):
    '''Seconds for Nvidia 3090 GPU'''
    t = (sequence_length/29)**2	# For 3 recycles and 5 models, 1 prediction for each model
    t *= (recycles+1) / 4
    t *= num_models / 5
    return t

# -----------------------------------------------------------------------------
#
def _seconds_to_day_hour_minute(seconds, precision = 0.04):
    t = []
    s = seconds
    for unit, sec in (('day',86400), ('hour',3600), ('minute',60), ('second',1)):
        nu = int(s / sec)
        if t or nu > 0:
            t.append(f'{nu} {unit}{_plural(nu)}')
        s -= nu * sec
        if s / seconds < precision:
            break
    return ' '.join(t)

# -----------------------------------------------------------------------------
#
def _prediction_info(seqs, with_seqs, seq_pairs, long_pairs, recycles, models):
    pair_lengths = _sequence_pair_lengths(seq_pairs)
    rt = sum([_estimated_runtime(slen, recycles, len(models)) for slen in pair_lengths])
    rt_string = _seconds_to_day_hour_minute(rt)

    pmin_length, pmax_length = min(pair_lengths), max(pair_lengths)
    msg = f'{len(seq_pairs)} dimers with lengths {pmin_length}-{pmax_length}.'
    msg += f'\nEstimated prediction time {rt_string} using Nvidia 3090 GPU.'
    if long_pairs:
        long_lengths = _sequence_pair_lengths(long_pairs)
        lmin_length, lmax_length = min(long_lengths), max(long_lengths)
        msg += f'\nOmitted {len(long_pairs)} dimers with long sequence lengths {lmin_length}-{lmax_length}'
    mdescrip = _monomers_description(seqs, with_seqs)
    msg += f'\n{mdescrip}'
    return msg

# -----------------------------------------------------------------------------
#
def _monomers_description(seqs, with_seqs = None):
    sequences = sorted(seqs)
    seq_lengths = [len(seq) for name, seq in sequences]
    smin_length, smax_length = min(seq_lengths), max(seq_lengths)
    seq_names = ", ".join(f'{name} ({len(seq)})' for name, seq in sequences)
    msg = f'{len(seqs)} monomer sequences with lengths {smin_length}-{smax_length}:'
    if with_seqs is not None:
        with_sequences = sorted(with_seqs)
        wseq_lengths = [len(seq) for name, seq in with_sequences]
        wmin_length, wmax_length = min(wseq_lengths), max(wseq_lengths)
        msg += f' with {len(with_seqs)} sequences, lengths {wmin_length}-{wmax_length}'
        seq_names += " and " + ", ".join(f'{name} ({len(seq)})' for name, seq in with_sequences)
    msg += f'\n{seq_names}'
    return msg

# -----------------------------------------------------------------------------
#
def _plural(count):
    return 's' if count != 1 else ''

# -----------------------------------------------------------------------------
#
from chimerax.core.commands import Annotation, AnnotationError

class NamedSeqsArg(Annotation):
    '''
    Return list of pairs of sequence name and sequence.
    Sequences can be specified as a fasta file path, or an atom-spec for open chains,
    or a comma-separated list of uniprot names or identifiers.
    '''
    name = 'sequences'
        
    @classmethod
    def parse(cls, text, session):
        if len(text.strip()) == 0:
            raise AnnotationError('Missing sequences argument.')
        from chimerax.core.commands import next_token
        token, used, rest = next_token(text)
        import os.path
        path = os.path.expanduser(token)
        if path.endswith('.fasta') and os.path.exists(path):
            named_seqs = _read_fasta_sequences(path)
        else:
            from chimerax.atomic.args import is_atom_spec
            if is_atom_spec(text, session):
                from chimerax.atomic import UniqueChainsArg, Residue
                chains, used, rest = UniqueChainsArg.parse(text, session)
                chains = [c for c in chains if c.polymer_type == Residue.PT_AMINO]
                if len(chains) == 0:
                    raise AnnotationError('Sequences atom specifier includes no protein chains.')
                named_seqs = [(_chain_sequence_name(chain), chain.characters) for chain in chains]
            else:
                raise AnnotationError('Sequences argument must be a FASTA file path (suffix .fasta)'
                                      ' or a chain specifier.')
        return named_seqs, used, rest

# -----------------------------------------------------------------------------
#
def _read_fasta_sequences(path):
    f = open(path, 'r')
    named_seqs = []
    title = ''
    lines = []
    for line in f.readlines():
        if line.startswith('>'):
            if lines:
                named_seqs.append((title, ''.join(lines)))
            title = line[1:].strip()
            lines = []
        else:
            lines.append(line.strip())
    if lines:
        named_seqs.append((title, ''.join(lines)))
    return named_seqs

# -----------------------------------------------------------------------------
#
def _chain_sequence_name(chain):
    from chimerax.atomic import uniprot_ids
    uids = uniprot_ids(chain.structure)
    for uid in uids:
        if uid.chain_id == chain.chain_id:
            if uid.uniprot_name:
                return uid.uniprot_name
            elif uid.uniprot_id:
                return uid.uniprot_id
    # return chain.description
    return f'{chain.structure.name}_{chain.chain_id}'

# -----------------------------------------------------------------------------
#
def alphafold_monomers(session, sequences = None, max_length = None, recycles = 3,
                       models = (1,2,3,4,5), output_fasta = None, output_json = None,
                       open = '.'):
    '''
    Estimate time for predicting a set of monomers using
    localcolabfold (https://github.com/YoshitakaMo/localcolabfold).
    '''
    if sequences is None:
        alphafold_open_monomers(session, open)
        return
    
    from chimerax.core.errors import UserError
    if len(sequences) == 0:
        raise UserError('No sequences specified')

    useqs = _unique_sequences(sequences)
    seqs = [(name,seq) for name,seq in useqs if len(seq) <= max_length] if max_length else useqs

    rt = sum([_estimated_runtime(len(seq), recycles, len(models)) for name,seq in seqs])
    rt_string = _seconds_to_day_hour_minute(rt)
    msg = f'Estimated prediction time {rt_string} using Nvidia 3090 GPU.'

    sdescrip = _monomers_description(seqs)
    msg += f'\n{sdescrip}'

    if len(seqs) < len(useqs):
        lseqs = [(name,seq) for name,seq in useqs if len(seq) > max_length]
        ldescrip = _monomers_description(lseqs)
        msg += f'\nOmitted {ldescrip}'

    seq_path = 'sequences.fasta' if output_fasta is None else output_fasta
    cmd = colabfold_batch_command(seq_path, recycles, models)
    msg += f'\nPrediction command: {cmd}'

    session.logger.info(msg)

    if output_fasta:
        write_monomers_fasta(output_fasta, seqs)

    if output_json:
        write_monomers_json(output_json, seqs)
        
# -----------------------------------------------------------------------------
#
def alphafold_open_monomers(session, directory = '.'):
    file_pattern = '*rank_001*.pdb'
    if directory != '.':
        from os.path import join
        file_pattern = join(directory, file_pattern)

    from chimerax.core.commands import run
    models = run(session, f'open {file_pattern} logInfo false')

    if len(models) == 0:
        return models

    # Rename models to short name
    for m in models:
        i = m.name.find('_unrelaxed')
        if i >= 0:
            m.name = m.name[:i]

    from chimerax.core.commands import concise_model_spec
    mspec = concise_model_spec(session, models, allow_empty_spec = False)
    run(session, f'label {mspec} models')
    run(session, f'tile {mspec}')
    run(session, f'color bfactor {mspec} palette alphafold')

    return models
    
# -----------------------------------------------------------------------------
#
def register_alphafold_monomers_command(logger):
    from chimerax.core.commands import CmdDesc, register, IntArg, IntsArg, SaveFileNameArg, OpenFolderNameArg
    desc = CmdDesc(
        optional = [('sequences', NamedSeqsArg)],
        keyword = [('max_length', IntArg),
                   ('recycles', IntArg),
                   ('models', IntsArg),
                   ('output_fasta', SaveFileNameArg),
                   ('output_json', SaveFileNameArg),
                   ('open', OpenFolderNameArg),
                   ],
        synopsis = 'Estimate runtime for monomer AlphaFold predictions using colabfold_batch'
    )
    from chimerax.core.commands import register
    register('alphafold monomers', desc, alphafold_monomers, logger=logger)

# -----------------------------------------------------------------------------
#
def register_alphafold_dimers_command(logger):
    from chimerax.core.commands import CmdDesc, register, IntArg, IntsArg, BoolArg, SaveFileNameArg
    desc = CmdDesc(
        required = [('sequences', NamedSeqsArg)],
        keyword = [('with_sequences', NamedSeqsArg),
                   ('homodimers', BoolArg),
                   ('max_length', IntArg),
                   ('recycles', IntArg),
                   ('models', IntsArg),
                   ('output_fasta', SaveFileNameArg),
                   ('output_json', SaveFileNameArg),
                   ],
        synopsis = 'Setup AlphaFold predictions for all dimers for a set of sequences using colabfold_batch'
    )
    from chimerax.core.commands import register
    register('alphafold dimers', desc, alphafold_dimers, logger=logger)
