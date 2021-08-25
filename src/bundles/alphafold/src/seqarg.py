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
#
from chimerax.core.commands import Annotation, AnnotationError
class SequencesArg(Annotation):
    '''
    Accept a chain atom spec (#1/A), a sequence viewer alignment id (myseqs.aln:2),
    a UniProt accession id (K9Z9J3, 6 or 10 characters, always has numbers),
    a UniProt name (MYOM1_HUMAN, always has underscore, X_Y where X and Y are at most
    5 alphanumeric characters), or a sequence (MVLSPADKTN....).
    Returns a list of Sequence objects or Sequence subclass objects such as Chains.
    '''
    name = 'sequences'
    
    @classmethod
    def parse(cls, text, session):
        if is_atom_spec(text, session):
            from chimerax.atomic import UniqueChainsArg
            return UniqueChainsArg.parse(text, session)
        elif is_uniprot_id(text):
            value, used, rest = UniProtSequenceArg.parse(text, session)
            return [value], used, rest
        else:
            for argtype in (AlignmentSequenceArg, RawSequenceArg):
                try:
                    value, used, rest = argtype.parse(text, session)
                    return [value], used, rest
                except Exception:
                    continue
        from chimerax.core.commands import next_token
        token, text, rest = next_token(text)
        raise AnnotationError('Sequences argument "%s" is not a chain specifier, ' % token +
                              'alignment id, UniProt id, or sequence characters')

class SequenceArg(Annotation):
    name = 'sequence'
    
    @classmethod
    def parse(cls, text, session):
        value, used, rest = SequencesArg.parse(text, session)
        if len(value) != 1:
            raise AnnotationError('Sequences argument "%s" must specify 1 sequence, got %d'
                                  % (used, len(value)))
        return value[0], used, rest
    
def is_atom_spec(text, session):
    from chimerax.core.commands import AtomSpecArg
    try:
        AtomSpecArg.parse(text, session)
    except AnnotationError:
        return False
    return True

def is_uniprot_id(text):
    # Name and accession format described here.
    # https://www.uniprot.org/help/accession_numbers
    # https://www.uniprot.org/help/entry_name
    from chimerax.core.commands import next_token
    id, text, rest = next_token(text)
    if '_' in id:
        fields = id.split('_')
        f1,f2 = fields
        if (f1.isalnum() and len(f1) <= 6 or len(f1) == 10 and
            f2.isalnum() and len(f2) <= 5):
            return True
    elif id.isalnum() and id[0].isalpha() and id[1].isdigit and id[5].isdigit():
        if len(id) == 6:
            return True
        elif len(id) == 10 and id[6].isalpha() and id[9].isdigit():
            return True
    return False
                
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
        if len(uid) not in (6, 10):
            raise AnnotationError('UniProt id "%s" must be 6 or 10 characters' % uid)
        from chimerax.uniprot.fetch_uniprot import fetch_uniprot_accession_info
        try:
            seq_string, full_name, features = fetch_uniprot_accession_info(session, uid)
        except Exception:
            raise AnnotationError('Failed getting sequence for UniProt id "%s"' % uid)
        from chimerax.atomic import Sequence
        seq = Sequence(name = (uname or uid), characters = seq_string)
        seq.uniprot_accession = uid
        if uname is not None:
            seq.uniprot_name = uname
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
