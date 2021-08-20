# vim: set expandtab ts=4 sw=4:

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

# -----------------------------------------------------------------------------
# Search AlphaFold database for sequences using BLAST
#
from chimerax.core.commands import Annotation, AnnotationError
class SequenceArg(Annotation):
    '''
    Accept a chain atom spec (#1/A), a sequence viewer alignment id (myseqs.aln:2),
    a UniProt accession id (K9Z9J3, 6 or 10 characters, always has numbers),
    a UniProt name (MYOM1_HUMAN, always has underscore, X_Y where X and Y are at most
    5 alphanumeric characters), or a sequence (MVLSPADKTN....).
    Returns a Sequence object or a subclass such as Chain.
    '''
    name = 'sequence'
    
    @classmethod
    def parse(cls, text, session):
        from chimerax.atomic import ChainArg
        for argtype in (ChainArg, AlignmentSequenceArg, UniProtSequenceArg, RawSequenceArg):
            try:
                return argtype.parse(text, session)
            except Exception:
                continue
        from chimerax.core.commands import next_token
        token, text, rest = next_token(text)
        raise AnnotationError('Sequence argument "%s" is not a chain specifier, ' % token +
                              'alignment id, UniProt id, or sequence characters')

class AlignmentSequenceArg(Annotation):
    name = 'alignment sequence'
    
    @classmethod
    def parse(cls, text, session):
        from chimerax.seqalign import AlignSeqPairArg
        (alignment, seq), used, rest = AlignSeqPairArg.parse(text, session)
        return seq, used, rest

class UniProtSequenceArg(Annotation):
    name = 'UniProt sequence'
    
    @classmethod
    def parse(cls, text, session):
        from chimerax.core.commands import StringArg
        uid, used, rest = StringArg.parse(text, session)
        if '_' in uid:
            uname = uid
            from chimerax.uniprot import map_uniprot_ident
            try:
                uid = map_uniprot_ident(uid, return_value = 'entry')
            except Exception:
                raise AnnotationError('UniProt name "%s" must be 1-5 characters followed by an underscore followed by 1-5 characters' % uid)
        else:
            uname = None
        print ('uniprot id', uid, len(uid))
        if len(uid) not in (6, 10):
            raise AnnotationError('UniProt id "%s" must be 6 or 10 characters' % uid)
        from chimerax.uniprot.fetch_uniprot import fetch_uniprot_accession_info
        try:
            seq_string, full_name, features = fetch_uniprot_accession_info(session, uid)
        except Exception:
            raise AnnotationError('Failed getting sequence for UniProt id "%s"' % uid)
        from chimerax.atomic import Sequence
        seq = Sequence(name = (uname or uid), characters = seq_string)
        return seq, used, rest

class RawSequenceArg(Annotation):
    name = 'raw sequence'
    
    @classmethod
    def parse(cls, text, session):
        from chimerax.core.commands import StringArg
        seqchars, used, rest = StringArg.parse(text, session)
        upper_a_to_z = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        if not set(seqchars).issubset(upper_a_to_z):
            nonalpha = ''.join(set(seqchars) - upper_a_to_z)
            raise AnnotationError('Sequence "%s" contains characters "%s" that are not upper case A to Z.'
                                  % (seqchars, nonalpha))
        from chimerax.atomic import Sequence
        seq = Sequence(characters = seqchars)
        return seq, used, rest
