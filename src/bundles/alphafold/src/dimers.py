# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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
def alphafold_dimers(session, sequences, with_sequences = None, complex = None,
                     homodimers = True, max_length = None, recycles = 3,
                     networks = (1,2,3,4,5), gpu = None, output_fasta = None,
                     directory = None, run = True):
    '''
    Run predictions of all combinations of dimers for a given set of sequences
    using localcolabfold (https://github.com/YoshitakaMo/localcolabfold) installed on the same computer.
    Analyze the resulting predictions to protein-protein interactions which AlphaFold gives
    high confidence.  Visualize those interfaces.
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

    msg = _prediction_info(seqs, with_seqs, seq_pairs, long_pairs, recycles, networks)
    session.logger.info(msg)

    if output_fasta:
        write_dimers_fasta(output_fasta, seq_pairs)
        
# -----------------------------------------------------------------------------
#
def run_predictions(fasta_path, recycles, networks, gpu, directory):
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
def _estimated_runtime(sequence_length, recycles = 3, networks = 5):
    '''Seconds for Nvidia 3090 GPU'''
    t = (sequence_length/29)**2	# For 3 recycles and 5 networks, 1 model each network
    t *= (recycles+1) / 4
    t *= networks / 5
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
def _prediction_info(seqs, with_seqs, seq_pairs, long_pairs, recycles, networks):
    pair_lengths = _sequence_pair_lengths(seq_pairs)
    rt = sum([_estimated_runtime(slen, recycles, len(networks)) for slen in pair_lengths])
    rt_string = _seconds_to_day_hour_minute(rt)

    pmin_length, pmax_length = min(pair_lengths), max(pair_lengths)
    msg = f'Predicting {len(seq_pairs)} dimers with lengths {pmin_length}-{pmax_length}'
    msg += f',\nestimated run time {rt_string} using Nvidia 3090 GPU'
    if long_pairs:
        long_lengths = _sequence_pair_lengths(long_pairs)
        lmin_length, lmax_length = min(long_lengths), max(long_lengths)
        msg += f',\nomitting {len(long_pairs)} dimers with long sequence lengths {lmin_length}-{lmax_length}'
    mdescrip = _monomers_description(seqs, with_seqs)
    msg += f',\n{mdescrip}'
    return msg

# -----------------------------------------------------------------------------
#
def _monomers_description(seqs, with_seqs):
    seq_lengths = [len(seq) for name, seq in seqs]
    smin_length, smax_length = min(seq_lengths), max(seq_lengths)
    seq_names = ", ".join(f'{name} ({len(seq)})' for name, seq in seqs)
    msg = f'dimers from {len(seqs)} monomer sequences with lengths {smin_length}-{smax_length}'
    if with_seqs is not None:
        wseq_lengths = [len(seq) for name, seq in with_seqs]
        wmin_length, wmax_length = min(wseq_lengths), max(wseq_lengths)
        msg += f' with {len(with_seqs)} sequences, lengths {wmin_length}-{wmax_length}'
        seq_names += " and " + ", ".join(f'{name} ({len(seq)})' for name, seq in with_seqs)
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
    return f'{chain.structure.name} {chain.chain_id}'

# -----------------------------------------------------------------------------
#
def dimers_command_description():
    from chimerax.core.commands import CmdDesc, register, IntArg, IntsArg, BoolArg, SaveFileNameArg
    desc = CmdDesc(
        required = [('sequences', NamedSeqsArg)],
        keyword = [('with_sequences', NamedSeqsArg),
                   ('complex', NamedSeqsArg),
                   ('homodimers', BoolArg),
                   ('max_length', IntArg),
                   ('recycles', IntArg),
                   ('networks', IntsArg),
                   ('output_fasta', SaveFileNameArg),
                   ('gpu', IntArg),
                   ('run', BoolArg),
                   ],
        synopsis = 'Run AlphaFold predictions for all dimers for a set of sequences using colabfold_batch installed on the local computer'
    )
    return desc
    
# -----------------------------------------------------------------------------
#
def register_alphafold_dimers_command(logger):
    desc = dimers_command_description()
    from chimerax.core.commands import register
    register('alphafold dimers', desc, alphafold_dimers, logger=logger)
