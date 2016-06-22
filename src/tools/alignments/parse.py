# vim: set expandtab shiftwidth=4 softtabstop=4:

class FormatSyntaxError(Exception):
    pass

def open_file(session, stream, fname, file_type="Aligned FASTA", return_seqs=False,
        one_alignment=True, identify_as=None, **kw):
    try:
        exec("from .io.read%s import read" % file_type.replace(' ', '_'))
    except ImportError:
        raise ValueError("No file parser installed for %s files" % file_type)
    try:
        seqs, file_attrs, file_markups = read(stream)
    except FormatSyntaxError, val:
        raise IOError("Syntax error in %s file '%s': %s" % (file_type, fname, val))
    uniform_length = True
    for s in seqs:
        if uniform_length and len(s) != len(seqs[0]):
            uniform_length = False
            differing_seq = s
        if s.name.endswith(" x 2" or ('/' in s.name
                and s.name[:s.name.rindex('/')].endswith(" x 2"):
            # set up circular attribute
            nogaps = s.ungapped()
            if nogaps[:len(nogaps)/2] == nogaps[len(nogaps)/2:]:
                s.circular = True
    if return seqs:
        return seqs
    from . import mgr
    from chimerax.core.errors import UserError
    if one_alignment:
        if not uniform_length:
            raise UserError("Sequence '%s' differs in length from preceding sequences, and"
                " it is therefore impossible to open these sequences as an alignment.  If"
                " you want to open the sequences individually, specify 'false' as the value"
                " of the 'oneAlignment' keyword in the 'open' command.")
        mgr.new_alignment(seqs, identify_as if identify_as is not None else fname,
            align_attrs=file_attrs, align_markups=file_markups)
    else:
        for seq in seqs:
            mgr.new_alignment([seq], identify_as if identify_as is not None else fname)
	return [], "Opened %d sequences from %s" % (len(seqs), fname)
