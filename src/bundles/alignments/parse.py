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

class FormatSyntaxError(Exception):
    pass

def open_file(session, stream, fname, format_name="FASTA", return_vals=None,
        one_alignment=True, identify_as=None, auto_associate=True, **kw):
    ns = {}
    try:
        exec("from .io.read%s import read" % format_name.replace(' ', '_'), globals(), ns)
    except ImportError:
        raise ValueError("No file parser installed for %s files" % format_name)
    if stream is None:
        import os.path
        from chimerax.core.io import open_filename
        path = fname
        fname = os.path.basename(path)
        stream = open_filename(path)
    try:
        seqs, file_attrs, file_markups = ns['read'](stream)
    except FormatSyntaxError as err:
        raise IOError("Syntax error in %s file '%s': %s" % (format_name, fname, err))
    if not seqs:
        raise ValueError("No sequences found in %s file '%s'!" % (format_name, fname))
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
    if return_vals == "seqs":
        return seqs
    from chimerax.core.errors import UserError
    if one_alignment:
        if not uniform_length:
            raise UserError("Sequence '%s' differs in length from preceding sequences, and"
                " it is therefore impossible to open these sequences as an alignment.  If"
                " you want to open the sequences individually, specify 'false' as the value"
                " of the 'oneAlignment' keyword in the 'open' command." % differing_seq.name)
        alignments = [session.alignments.new_alignment(seqs,
            identify_as if identify_as is not None else fname, align_attrs=file_attrs,
            align_markups=file_markups, auto_associate=auto_associate, **kw)]
    else:
        alignments = []
        for seq in seqs:
            alignments.append(session.alignments.new_alignment([seq],
                identify_as if identify_as is not None else fname,
                auto_associate=auto_associate, **kw))
    if return_vals == "alignments":
        return alignments
    return [], "Opened %d sequences from %s" % (len(seqs), fname)

def make_readable(seq_name):
    """Make sequence name more human-readable"""
    return seq_name.strip()
