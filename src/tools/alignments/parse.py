# vim: set expandtab shiftwidth=4 softtabstop=4:

class FormatSyntaxError(Exception):
    pass

def open_file(session, stream, fname, file_type="FASTA", return_seqs=False,
        one_alignment=True, identify_as=None, **kw):
    ns = {}
    try:
        exec("from .io.read%s import read" % file_type.replace(' ', '_'), globals(), ns)
    except ImportError:
        raise ValueError("No file parser installed for %s files" % file_type)
    # don't want the binary stream (e.g. "line[0] == '>'" is always False(!))
    path = stream.name
    stream.close()
    from chimerax.core.io import open_filename
    with open_filename(path) as f:
        try:
            seqs, file_attrs, file_markups = ns['read'](f)
        except FormatSyntaxError as err:
            raise IOError("Syntax error in %s file '%s': %s" % (file_type, fname, err))
    uniform_length = True
    for s in seqs:
        if uniform_length and len(s) != len(seqs[0]):
            uniform_length = False
            differing_seq = s
        if s.name.endswith(" x 2") or ('/' in s.name
                and s.name[:s.name.rindex('/')].endswith(" x 2")):
            # set up circular attribute
            nogaps = s.ungapped()
            if nogaps[:len(nogaps)/2] == nogaps[len(nogaps)/2:]:
                s.circular = True
    if return_seqs:
        return seqs
    from chimerax.core.errors import UserError
    if one_alignment:
        if not uniform_length:
            raise UserError("Sequence '%s' differs in length from preceding sequences, and"
                " it is therefore impossible to open these sequences as an alignment.  If"
                " you want to open the sequences individually, specify 'false' as the value"
                " of the 'oneAlignment' keyword in the 'open' command.")
        session.alignments.new_alignment(seqs, identify_as if identify_as is not None else fname,
            align_attrs=file_attrs, align_markups=file_markups)
    else:
        for seq in seqs:
            session.alignments.new_alignment([seq],
                identify_as if identify_as is not None else fname)
    return [], "Opened %d sequences from %s" % (len(seqs), fname)

def make_readable(seq_name):
    """Make sequence name more human-readable"""
    return seq_name.strip()
