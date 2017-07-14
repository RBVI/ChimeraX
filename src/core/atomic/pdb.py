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

"""
pdb: PDB format support
=======================

Read Protein DataBank (PDB) files.
"""

def open_pdb(session, stream, file_name, auto_style=True, coordsets=False):

    path = stream.name if hasattr(stream, 'name') else None

    from . import pdbio
    pointers = pdbio.read_pdb_file(stream, log=session.logger, explode=not coordsets)
    stream.close()

    from .structure import AtomicStructure
    models = [AtomicStructure(session, name = file_name, c_pointer = p, auto_style = auto_style) for p in pointers]

    if path:
        for m in models:
            m.filename = path

    info = "Opened PDB data containing %d atoms%s %d bonds" % (
        sum(m.num_atoms for m in models),
        ("," if coordsets else " and"),
        sum(m.num_bonds for m in models))
    if coordsets:
        num_cs = 0
        for m in models:
            num_cs += m.num_coordsets
        info += " and %s coordinate sets" % num_cs
        if session.ui.is_gui:
            mc = [m for m in models if m.num_coordsets > 1]
            if mc:
                from ..commands.coordset import coordset_slider
                coordset_slider(session, mc)

    return models, info


def save_pdb(session, path, format, models=None, selected_only=False, displayed_only=False,
        all_frames=False, pqr=False, rel_model=None):
    from ..errors import UserError
    if models is None:
        from . import Structure
        models = [m for m in session.models if isinstance(m, Structure)]
    if not models:
        raise UserError("No structures to save")

    if len(models) == 1 and list(models)[0] == rel_model:
        xforms = None
    else:
        xforms = []
        for s in models:
            if rel_model is None:
                xforms.append(s.scene_position.matrix)
            else:
                inv = rel_model.scene_position.inverse()
                if s.scene_position == rel_model.scene_position:
                    xforms.append(None)
                else:
                    xforms.append((s.scene_position * inv).matrix)

    from . import pdbio
    file_per_model = "[NAME]" in path or "[NUMBER]" in path
    if file_per_model:
        for m, xform in zip(models, xforms):
            file_name = path.replace("[NUMBER]", m.id_string()).replace("[NAME]", m.name)
            pdbio.write_pdb_file([m.cpp_pointer], file_name, selected_only=selected_only,
                displayed_only=displayed_only, xforms=[xform], all_frames=all_frames, pqr=pqr)
    else:
        pdbio.write_pdb_file([m.cpp_pointer for m in models], path, selected_only=selected_only,
            displayed_only=displayed_only, xforms=xforms, all_frames=all_frames, pqr=pqr)


_pdb_sources = {
    "rcsb": "http://www.pdb.org/pdb/files/%s.pdb",
    "pdbe": "http://www.ebi.ac.uk/pdbe/entry-files/download/pdb%s.ent",
    # "pdbj": "https://pdbj.org/rest/downloadPDBfile?format=pdb&id=%s",
}


def fetch_pdb(session, pdb_id, fetch_source="rcsb", ignore_cache=False, **kw):
    if len(pdb_id) != 4:
        from ..errors import UserError
        raise UserError('PDB identifiers are 4 characters long, got "%s"' % pdb_id)
    import os
    pdb_id = pdb_id.lower()
    # check on local system -- TODO: configure location
    subdir = pdb_id[1:3]
    filename = "/databases/mol/pdb/%s/pdb%s.ent" % (subdir, pdb_id)
    if os.path.exists(filename):
        session.logger.info("Fetching PDB %s from system cache: %s" % (pdb_id, filename))
    else:
        base_url = _pdb_sources.get(fetch_source, None)
        if base_url is None:
            raise UserError('unrecognized PDB source "%s"' % fetch_source)
        url = base_url % pdb_id
        pdb_name = "%s.pdb" % pdb_id
        from ..fetch import fetch_file
        filename = fetch_file(session, url, 'PDB %s' % pdb_id, pdb_name, 'PDB',
                              ignore_cache=ignore_cache)

    session.logger.status("Opening PDB %s" % (pdb_id,))
    from .. import io
    models, status = io.open_data(session, filename, format='pdb', name=pdb_id, **kw)
    return models, status



def fetch_pdb_pdbe(session, pdb_id, **kw):
    return fetch_pdb(session, pdb_id, fetch_source="pdbe", **kw)


def fetch_pdb_pdbj(session, pdb_id, **kw):
    return fetch_pdb(session, pdb_id, fetch_source="pdbj", **kw)


def register_pdb_format():
    from .. import io
    from . import structure
    io.register_format(
        "PDB", structure.CATEGORY, (".pdb", ".pdb1", ".ent", ".pqr"), ("pdb",),
        mime=("chemical/x-pdb", "chemical/x-spdbv"),
        reference="http://wwpdb.org/docs.html#format",
        open_func=open_pdb, export_func=save_pdb)
    from ..commands import add_keyword_arguments, BoolArg, StructuresArg, ModelArg
    add_keyword_arguments('open', {'coordsets':BoolArg, 'auto_style':BoolArg})
    add_keyword_arguments('save', {'models':StructuresArg, 'selected_only':BoolArg,
        'displayed_only':BoolArg, 'all_frames':BoolArg, 'pqr':BoolArg, 'rel_model':ModelArg})


def register_pdb_fetch():
    from .. import fetch
    fetch.register_fetch('pdb', fetch_pdb, 'pdb', prefixes = [])
    fetch.register_fetch('pdbe', fetch_pdb_pdbe, 'pdb', prefixes = [])
    # PDBj is unreliable for PDB format, mmCIF seemed okay - CH 2dec16
    # fetch.register_fetch('pdbj', fetch_pdb_pdbj, 'pdb', prefixes = [])


def process_chem_name(name, use_greek=True, probable_abbrs=False):
    if name.isupper() and (" " in name or len(name) > 5):
        # probable non-abbreviation all uppercase:  need to downcase as appropriate
        text = ""
        word = ""
        for c in name.strip().lower():
            if c.isalpha():
                word += c
                continue
            if word:
                if c.isdigit() or (text and text[-1].isdigit()):
                    text += _process_chem_word(word, use_greek, probable_abbrs).upper()
                else:
                    text += _process_chem_word(word, use_greek, probable_abbrs)
                word = ""
            text += c
        if word:
            if c.isdigit() or (text and text[-1].isdigit()):
                text += _process_chem_word(word, use_greek, probable_abbrs).upper()
            else:
                text += _process_chem_word(word, use_greek, probable_abbrs)
    else:
        # already mixed case; just substitute greek letters as appropriate
        if use_greek:
            processed_words = []
            for word in name.split():
                if word.lower() in greek_letters:
                    processed_words.append(greek_letters[word.lower()])
                else:
                    processed_words.append(word)
            text = " ".join(processed_words)
        else:
            text = name
    return text

greek_letters = {
    'alpha': u'\N{GREEK SMALL LETTER ALPHA}',
    'beta': u'\N{GREEK SMALL LETTER BETA}',
    'gamma': u'\N{GREEK SMALL LETTER GAMMA}',
    'delta': u'\N{GREEK SMALL LETTER DELTA}',
    'epsilon': u'\N{GREEK SMALL LETTER EPSILON}',
    'zeta': u'\N{GREEK SMALL LETTER ZETA}',
    'eta': u'\N{GREEK SMALL LETTER ETA}',
    'theta': u'\N{GREEK SMALL LETTER THETA}',
    'iota': u'\N{GREEK SMALL LETTER IOTA}',
    'kappa': u'\N{GREEK SMALL LETTER KAPPA}',
    'lambda': u'\N{GREEK SMALL LETTER LAMDA}',
    'lamda': u'\N{GREEK SMALL LETTER LAMDA}',
    'mu': u'\N{GREEK SMALL LETTER MU}',
    'nu': u'\N{GREEK SMALL LETTER NU}',
    'xi': u'\N{GREEK SMALL LETTER XI}',
    'omicron': u'\N{GREEK SMALL LETTER OMICRON}',
    'pi': u'\N{GREEK SMALL LETTER PI}',
    'rho': u'\N{GREEK SMALL LETTER RHO}',
    'sigma': u'\N{GREEK SMALL LETTER SIGMA}',
    'tau': u'\N{GREEK SMALL LETTER TAU}',
    'upsilon': u'\N{GREEK SMALL LETTER UPSILON}',
    'phi': u'\N{GREEK SMALL LETTER PHI}',
    'chi': u'\N{GREEK SMALL LETTER CHI}',
    'psi': u'\N{GREEK SMALL LETTER PSI}',
    'omega': u'\N{GREEK SMALL LETTER OMEGA}',
}
def _process_chem_word(word, use_greek, probable_abbrs):
    if len(word) == 1:
        return word.upper()
    from . import Element
    if len(word) == 2 and word.capitalize() in Element.names:
        return word.capitalize()
    if set(list(word)) <= set(["i", "v", "x"]):
        # Roman numeral
        return word.upper()
    if use_greek and word in greek_letters:
        return greek_letters[word]
    if probable_abbrs and len(word) < 5 and word != "cell":
        return word.upper()
    segs = []
    for seg in word.split('-'):
        if seg.startswith("rna") or seg.startswith("dna"):
            segs.append(seg[:3].upper() + seg[3:])
        elif seg.endswith("rna") or seg.endswith("dna"):
            segs.append(seg[:-3] + seg[-3:].upper())
        else:
            segs.append(word)
    return '-'.join(segs)
