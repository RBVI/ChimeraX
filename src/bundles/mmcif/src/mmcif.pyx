# distutils: language=c++
# cython: language_level=3, boundscheck=False, auto_pickle=False
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
mmcif: mmCIF format support
===========================

Read mmCIF files.
"""

import sys
from chimerax.core.errors import UserError

_builtin_open = open
_initialized = False

_additional_categories = (
    "pdbx_struct_assembly",
    "pdbx_struct_assembly_gen",
    "pdbx_struct_oper_list",
    "entity",
    "entity_src_gen",
    "entity_src_nat",
    "cell",
    "symmetry",
    "struct_ncs_oper",
    "atom_sites",
    "software",
    "struct",
    "citation",
    "citation_author",
    "citation_editor",
    "chem_comp",
    "exptl",
    "refine",
    "reflns",
    "em_3d_reconstruction",
    "exptl",
)
# _reserved_words = {
#     'loop_', 'stop_', 'global_', "data_", "save_"
# }


def _initialize(session):
    global _initialized
    _initialized = True
    from os.path import join, dirname, exists
    from . import _mmcif
    std_residues = join(dirname(__file__), "stdresidues.cif")
    if exists(std_residues):
        _mmcif.load_mmCIF_templates(std_residues)
    _mmcif.set_Python_locate_function(
        lambda name, session=session: _get_template(session, name))


def open_mmcif(session, path, file_name=None, auto_style=True, coordsets=False, atomic=True,
               max_models=None, log_info=True, extra_categories=(), combine_sym_atoms=True):
    # mmCIF parsing requires an uncompressed file

    if not _initialized:
        _initialize(session)

    from . import _mmcif
    categories = _additional_categories + tuple(extra_categories)
    log = session.logger if log_info else None
    try:
        pointers = _mmcif.parse_mmCIF_file(path, categories, log, coordsets, atomic)
    except _mmcif.error as e:
        raise UserError('mmCIF parsing error: %s' % e)

    if file_name is None:
        from os.path import basename
        file_name = basename(path)
    if atomic:
        from chimerax.atomic.structure import AtomicStructure as StructureClass
    else:
        from chimerax.atomic.structure import Structure as StructureClass
    models = [StructureClass(session, name=file_name, c_pointer=p, auto_style=auto_style, log_info=log_info)
              for p in pointers]

    if max_models is not None:
        for m in models[max_models:]:
            m.delete()
        models = models[:max_models]

    for m in models:
        m.filename = path
        if combine_sym_atoms:
            m.combine_sym_atoms()

    info = ''
    if coordsets:
        num_cs = 0
        for m in models:
            num_cs += m.num_coordsets
        info = '%s has %d coordinate sets' % (file_name, num_cs)
        if session.ui.is_gui:
            mc = [m for m in models if m.num_coordsets > 1]
            if mc:
                from chimerax.std_commands.coordset import coordset_slider
                coordset_slider(session, mc)
    for model in models:
        model.is_mmcif = True	# Indicates metadata is from mmcif.
        struct = get_mmcif_tables_from_metadata(model, ["struct"])[0]
        if not struct:
            continue
        try:
            title = struct.fields(['title'])[0][0]
        except TableMissingFieldsError:
            continue
        from chimerax.pdb import process_chem_name
        model.html_title = process_chem_name(title, sentences=True)
        model.has_formatted_metadata = lambda ses: True
        # use proxy to avoid circular ref
        from weakref import proxy
        from types import MethodType
        model.get_formatted_metadata = MethodType(_get_formatted_metadata, proxy(model))
        model.get_formatted_res_info = MethodType(_get_formatted_res_info, proxy(model))
        break
    return models, info


def _get_formatted_metadata(model, session, *, verbose=False):
    from chimerax.core.logger import html_table_params
    from chimerax.pdb import process_chem_name
    html = "<table %s>\n" % html_table_params
    html += ' <thead>\n'
    html += '  <tr>\n'
    html += '   <th colspan="2">Metadata for %s</th>\n' % model
    html += '  </tr>\n'
    html += ' </thead>\n'
    html += ' <tbody>\n'

    # title
    if hasattr(model, 'html_title'):
        html += '  <tr>\n'
        html += '   <th>Title</th>\n'
        html += '   <td>%s</td>\n' % model.html_title
        html += '  </tr>\n'

    # citations
    cites = citations(model)
    if cites:
        html += '  <tr>\n'
        if len(cites) > 1:
            html += '   <th rowspan="%d">Citations</th>\n' % len(cites)
        else:
            html += '   <th>Citation</th>\n'
        html += '   <td>%s</td>\n' % cites[0]
        html += '  </tr>\n'
        for cite in cites[1:]:
            html += '  <tr>\n'
            html += '   <td>%s</td>\n' % cite
            html += '  </tr>\n'

    # non-standard residues
    html += model.get_formatted_res_info(standalone=False)

    metadata = model.metadata  # get once from C++ layer

    # source
    nat, gen = get_mmcif_tables_from_metadata(model, ["entity_src_nat", "entity_src_gen"], metadata=metadata)
    if nat:
        html += _process_src(nat, "Source%s (natural)", [
            'common_name', 'pdbx_organism_scientific',
            'genus', 'species', 'pdbx_ncbi_taxonomy_id'])
    if gen:
        html += _process_src(gen, "Gene source%s", [
            'gene_src_common_name', 'pdbx_gene_src_scientific_name', 'gene_src_genus',
            'gene_src_species', 'pdbx_gene_src_ncbi_taxonomy_id'])
        if verbose:
            html += _process_src(gen, "Host organism%s", [
                'host_org_common_name', 'pdbx_host_org_scientific_name', 'host_org_genus',
                'host_org_species', 'pdbx_host_org_ncbi_taxonomy_id'])

    # experimental method; resolution
    experiment = get_mmcif_tables_from_metadata(model, ["exptl"], metadata=metadata)[0]
    if experiment:
        method = substitute_none_for_unspecified(experiment.fields(
            ['method'], allow_missing_fields=True)[0])[0]
        if method:
            html += '  <tr>\n'
            html += '   <th>Experimental method</th>\n'
            html += '   <td>%s</td>\n' % process_chem_name(method, sentences=True)
            html += '  </tr>\n'
    res = None
    reflections = get_mmcif_tables_from_metadata(model, ["reflns"], metadata=metadata)[0]
    if reflections:
        res = substitute_none_for_unspecified(reflections.fields(
            ['d_resolution_high'], allow_missing_fields=True)[0])[0]
    if res is None:
        refine = get_mmcif_tables_from_metadata(model, ["refine"], metadata=metadata)[0]
        if refine:
            res = substitute_none_for_unspecified(refine.fields(
                ['ls_d_res_high'], allow_missing_fields=True)[0])[0]
    if res is None:
        em = get_mmcif_tables_from_metadata(model, ["em_3d_reconstruction"], metadata=metadata)[0]
        if em:
            res = substitute_none_for_unspecified(em.fields(
                ['resolution'], allow_missing_fields=True)[0])[0]
    if res is not None:
        html += '  <tr>\n'
        html += '   <th>Resolution</th>\n'
        html += '   <td>%s\N{ANGSTROM SIGN}</td>\n' % res
        html += '  </tr>\n'

    html += ' </tbody>\n'
    html += "</table>"

    return html


def _get_formatted_res_info(model, *, standalone=True):
    def update_nonstd(model, nonstd_info):
        chem_comp = get_mmcif_tables_from_metadata(model, ["chem_comp"])[0]
        if chem_comp:
            raw_rows = chem_comp.fields(['id', 'name', 'pdbx_synonyms'], allow_missing_fields=True)
            for raw_row in raw_rows:
                if raw_row[0] not in nonstd_info:
                    continue
                row = substitute_none_for_unspecified(raw_row)
                if row[1] or row[2]:
                    nonstd_info[row[0]] = (row[0], row[1], row[2])
    from chimerax.pdb import format_nonstd_res_info
    return format_nonstd_res_info(model, update_nonstd, standalone)


def _process_src(src, caption, field_names):
    raw_rows = src.fields(field_names, allow_missing_fields=True)
    usable_rows = set()
    for raw_row in raw_rows:
        row = substitute_none_for_unspecified(raw_row)
        if row[:4] != [None, None, None, None]:
            usable_rows.add(tuple(row))
    html = ""
    if usable_rows:
        from chimerax.pdb.pdb import format_source_name
        rows = list(usable_rows)
        html += '  <tr>\n'
        if len(rows) > 1:
            html += '   <th rowspan="%d">%s</th>\n' % (len(rows), caption % 's')
        else:
            html += '   <th>%s</th>\n' % caption % ''
        html += '   <td>%s</td>\n' % format_source_name(*rows[0])
        html += '  </tr>\n'
        for row in rows[1:]:
            html += '  <tr>\n'
            html += '   <td>%s</td>\n' % format_source_name(*row)
            html += '  </tr>\n'
    return html


def substitute_none_for_unspecified(fields):
    substituted = []
    for field in fields:
        if field in ('?', '.', ''):
            substituted.append(None)
        else:
            substituted.append(field)
    return substituted


_mmcif_sources = {
    # "rcsb": "http://www.pdb.org/pdb/files/%s.cif",
    "rcsb": "http://files.rcsb.org/download/%s.cif",
    "pdbe": "http://www.ebi.ac.uk/pdbe/entry-files/download/%s.cif",
    "pdbe_updated": "http://www.ebi.ac.uk/pdbe/entry-files/download/%s_updated.cif",
    "pdbj": "https://pdbj.org/rest/downloadPDBfile?format=mmcif&id=%s",
}


def fetch_mmcif(
        session, pdb_id, fetch_source="rcsb", ignore_cache=False,
        structure_factors=False, over_sampling=1.5,  # for ChimeraX-Clipper plugin
        **kw):
    """Get mmCIF file by PDB identifier via the Internet"""
    if not _initialized:
        _initialize(session)

    if len(pdb_id) != 4:
        raise UserError('PDB identifiers are 4 characters long, got "%s"' % pdb_id)
    if structure_factors:
        try:
            from chimerax.clipper.io import fetch_cif
        except ImportError:
            raise UserError('Working with structure factors requires the '
                            'ChimeraX_Clipper plugin, available from the Tool Shed')

    import os
    pdb_id = pdb_id.lower()
    filename = None
    if not fetch_source.endswith('updated'):
        # check on local system -- TODO: configure location
        subdir = pdb_id[1:3]
        filename = "/databases/mol/mmCIF/%s/%s.cif" % (subdir, pdb_id)
        if os.path.exists(filename):
            session.logger.info("Fetching mmCIF %s from system cache: %s" % (pdb_id, filename))
        else:
            filename = None
        cache = 'PDB'
    else:
        cache = fetch_source
    if filename is None:
        base_url = _mmcif_sources.get(fetch_source, None)
        if base_url is None:
            raise UserError('unrecognized mmCIF/PDB source "%s"' % fetch_source)
        url = base_url % pdb_id
        pdb_name = "%s.cif" % pdb_id
        from chimerax.core.fetch import fetch_file
        filename = fetch_file(session, url, 'mmCIF %s' % pdb_id, pdb_name,
                              cache, ignore_cache=ignore_cache)
        # double check that a mmCIF file was downloaded instead of an
        # HTML error message saying the ID does not exist
        with open(filename, 'r') as f:
            line = f.readline()
            if not line.startswith(('data_', '#')):
                f.close()
                import os
                os.remove(filename)
                raise UserError("Invalid mmCIF identifier")

    session.logger.status("Opening mmCIF %s" % (pdb_id,))
    models, status = session.open_command.open_data(
        filename, format='mmcif', name=pdb_id, **kw)
    if structure_factors:
        sf_file = fetch_cif.fetch_structure_factors(
            session, pdb_id, fetch_source=fetch_source, ignore_cache=ignore_cache)
        from chimerax.clipper import get_map_mgr
        mmgr = get_map_mgr(models[0], create=True)
        if over_sampling < 1:
            warn_str = ('Map over-sampling rate cannot be less than 1. Resetting to 1.0')
            session.logger.warning(warn_str)
            over_sampling = 1
        mmgr.add_xmapset_from_file(sf_file, oversampling_rate=over_sampling)
        return [mmgr.crystal_mgr], status
    return models, status


def fetch_mmcif_pdbe(session, pdb_id, **kw):
    return fetch_mmcif(session, pdb_id, fetch_source="pdbe", **kw)


def fetch_mmcif_pdbe_updated(session, pdb_id, **kw):
    return fetch_mmcif(session, pdb_id, fetch_source="pdbe_updated", **kw)


def fetch_mmcif_pdbj(session, pdb_id, **kw):
    return fetch_mmcif(session, pdb_id, fetch_source="pdbj", **kw)


def _get_template(session, name):
    """Get Chemical Component Dictionary (CCD) entry"""
    from chimerax.core.fetch import fetch_file
    filename = '%s.cif' % name
    url = "http://ligand-expo.rcsb.org/reports/%s/%s/%s.cif" % (name[0], name,
                                                                name)
    try:
        return fetch_file(session, url, 'CCD %s' % name, filename, 'CCD')
    except (UserError, OSError):
        return None


def find_template_residue(session, name):
    """Supported API. Lookup mmCIF component template residue.

    The component is fetched from the web if not already loaded"""
    if not _initialized:
        _initialize(session)
    from . import _mmcif
    return _mmcif.find_template_residue(name)


def load_mmCIF_templates(filename):
    """Supported API. Load mmCIF component templates from given file

    The file format should match the components.cif file from the PDB"""
    from . import _mmcif
    _mmcif.load_mmCIF_templates(filename)


# def quote(value, max_len=60):
#     """Return CIF 1.1 data value version of string"""
#     # max_len is for mimicing the output from the PDB (see #2230)
#     if isinstance(value, (int, float)):
#         return str(value)
#     cdef int sing_quote, dbl_quote, line_break, special, i
#     cdef Py_UCS4 ch
#     cdef str examine, s
#     s = str(value)
#     if len(s) == 0:
#         return '""'
# 
#     ch = s[0]
#     sing_quote = ch == "'"
#     dbl_quote = ch == '"'
#     line_break = ch == '\n'
#     special = ch in ' _$[;'  # True if empty string too
#     if not (special or sing_quote or dbl_quote or line_break):
#         if s.endswith('_'):
#             if len(s) < 8:
#                 cf = s.casefold()
#                 special = cf in _reserved_words
#         elif len(s) > 5 and s[4] == '_':
#             cf = s[0:4].casefold()
#             special = cf.startswith(('data', 'save'))
#     for i in range(1, len(s)):
#         examine = s[i:i + 2]
#         if len(examine) == 2:
#             if examine[0] == '"':
#                 if examine[1].isspace():
#                     dbl_quote = True
#                 else:
#                     special = True
#                 continue
#             elif examine[0] == "'":
#                 if examine[1].isspace():
#                     sing_quote = True
#                 else:
#                     special = True
#                 continue
#         if examine[0].isspace():
#             if examine[0] == '\n':
#                 line_break = True
#             else:
#                 special = True
#     if line_break or (sing_quote and dbl_quote) or (max_len and len(s) > max_len):
#         return f'\n;{s}\n;\n'
#     if sing_quote:
#         return f'"{s}"'
#     if dbl_quote:
#         return f"'{s}'"
#     if special:
#         return f'"{s}"'
#     return s

from ._mmcif import quote_value as quote


def citations(model, only=None):
    """Return APA-style HTML citations for model

    Parameters
    ----------
    model : instance of a :py:class:`~chimerax.atomic.AtomicStructure`
        The model.
    """
    if only is not None:
        only = only.casefold()
    from chimerax.atomic.structure import Structure
    if not isinstance(model, Structure):
        return ""
    citation, citation_author = get_mmcif_tables_from_metadata(model, [
        "citation", "citation_author"])
    if not citation:
        return ""

    try:
        cf = citation.fields([
            'id', 'title', 'journal_abbrev', 'journal_volume', 'year'])
    except TableMissingFieldsError:
        return ""   # missing fields
    cf2 = citation.fields([
        'page_first', 'page_last',
        'journal_issue', 'pdbx_database_id_PubMed', 'pdbx_database_id_DOI'],
        allow_missing_fields=True)
    try:
        caf = citation_author.fields(['citation_id', 'name', 'ordinal'])
        caf = tuple((ci, n, int(o)) for ci, n, o in caf)
    except (TableMissingFieldsError, AttributeError):
        caf = ()
    citations = []
    from html import escape
    from itertools import zip_longest
    for (i, t, ja, jv, y), (pf, pl, ji, pm, doi) in zip_longest(cf, cf2):
        if only is not None and i.casefold() != only:
            continue
        authors = [(n, o) for ci, n, o in caf if ci == i]
        authors.sort(key=lambda x: x[1])
        authors = ', '.join([n for n, o in authors])
        c = escape(authors)
        if c and c[-1] != '.':
            c += '. '
        c += ' (%s). %s' % (escape(y), escape(t))
        if c[-1] != '.':
            c += '.'
        if ja and ja not in '.?':
            c += ' <i>%s</i>' % escape(ja)
            if jv and jv not in '.?':
                c += ', <i>%s</i>' % escape(jv)
                if ji and ji not in '.?':
                    c += '(%s)' % escape(ji)
            if pf and pf not in '.?':
                c += ', %s' % escape(pf)
                if pl and pl not in '.?':
                    c += '-%s' % escape(pl)
        if pm and pm not in '.?':
            if c[-1] != '.':
                c += '.'
            p = escape(pm)
            c += ' PMID: <a href="http://www.ncbi.nlm.nih.gov/pubmed/%s">%s</a>' % (p, p)
        if doi and doi not in '.?':
            if c[-1] != '.':
                c += '.'
            d = escape(doi)
            c += ' DOI: <a href="http://dx.doi.org/%s">%s</a>' % (d, d)
        citations.append(c)
    return citations


def add_citation(model, citation_id, info, authors=(), editors=(), *, metadata=None):
    """Add citation to model's mmCIF metadata

    Parameters
    ----------
    model : instance of a :py:class:`~chimerax.atomic.AtomicStructure`
        The model.
    citation_id : string
        The citation identifier.
    info : dictionary
        Information about citation
    authors : sequence
        Optional sequence of the authors.
    editors : sequence
        Optional sequence of book or book chapter editors.
    metadata : optional metadata dictonary
        Allow reuse of existing metadata dictionary.

    Update the mmCIF 'citation', 'citation_author', and 'citation_editor' tables
    with the given informaiton.  If the `citation_id` is already present in the
    citation table, then nothing is done.

    The `info` dictionary is for the relevant data items from the mmCIF citation category,
    http://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v40.dic/Categories/citation.html
    except for the citation `id` which is given as an argument to this function.
    The capitalization should match that in the mmCIF dictionary.
    In particular, the following data items are supported:
        'title', 'journal_abbrev', 'journal_volume', 'year',
        'page_first', 'page_last', 'journal_issue',
        'pdbx_database_id_PubMed', 'pdbx_database_id_DOI'
    """
    tables = _add_citation(model, citation_id, info, authors, editors, metadata=metadata)

    # update metadata
    for table in tables:
        if table is not None:
            table._set_metadata(model, metadata=metadata)


def _add_citation(model, citation_id, info, authors=(), editors=(),
                  *, metadata=None, return_existing=False):
    # do bulk of add_citation's work, but just return revised tables
    # and don't update metadata
    from chimerax.core.utils import flattened
    # get existing category tables
    citation, citation_author, citation_editor = get_mmcif_tables_from_metadata(
        model, ['citation', 'citation_author', 'citation_editor'], metadata=metadata)

    citation_id = str(citation_id)

    if citation is not None and citation.field_has('id', citation_id):
        if return_existing:
            return citation, citation_author, citation_editor
        return None, None, None

    # construct new table entries
    new_authors = CIFTable(
        "citation_author", [
            'citation_id', 'name', 'ordinal'
        ],
        flattened((citation_id, str(author), str(i))
                  for author, i in zip(authors, range(1, sys.maxsize))))
    new_editors = CIFTable(
        "citation_editor", [
            'citation_id', 'name', 'ordinal'
        ],
        flattened((citation_id, str(editor), str(i))
                  for editor, i in zip(editors, range(1, sys.maxsize))))
    possible_items = [
        # default data item ordering with correct capitalization
        'id', 'title', 'journal_abbrev', 'journal_volume',
        'page_first', 'page_last', 'journal_issue', 'year',
        'country',
        'book_id_ISBN',
        'book_title',
        'book_publisher',
        'book_publisher_city',
        'pdbx_database_id_PubMed', 'pdbx_database_id_DOI'
        'journal_id_ISSN',
        'journal_id_CSD',
        'journal_id_ASTM',
        'abstract_id_CAS',
        'database_id_CSD',
    ]
    citation_items = ['id']
    citation_data = [citation_id]
    cinfo = {k.casefold(): (k, str(v)) for k, v in info.items()}
    if 'id' in cinfo:
        del cinfo['id']
    for i in possible_items:
        ci = i.casefold()
        if ci not in cinfo:
            continue
        citation_items.append(i)
        citation_data.append(cinfo[ci][1])
        del cinfo[ci]
    for k, v in cinfo.values():
        citation_items.append(k)
        citation_data.append(v)
    new_citation = CIFTable("citation", citation_items, citation_data)

    # combine tables
    if new_authors.num_rows() > 0:
        if citation_author is None:
            citation_author = new_authors
        else:
            citation_author.extend(new_authors)
    if new_editors.num_rows() > 0:
        if citation_editor is None:
            citation_editor = new_editors
        else:
            citation_editor.extend(new_editors)
    if new_citation.num_rows() > 0:
        if citation is None:
            citation = new_citation
        else:
            citation.extend(new_citation)

    # return revised tables
    return (citation, citation_author, citation_editor)


def add_software(model, name, info, *, metadata=None):
    """Add citation to model's mmCIF metadata

    Parameters
    ----------
    model : instance of a :py:class:`~chimerax.atomic.AtomicStructure`
        The model.
    info : dictionary
        Information about software
    metadata : optional metadata dictonary
        Allow reuse of existing metadata dictionary.

    Update the mmCIF 'software' table with the given informaiton.  If the `name` is
    already present in the software table, then nothing is done.

    The `info` dictionary is for the relevant data items from the mmCIF softare category,
    http://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v40.dic/Categories/software.html
    except for the `name`, which is given as an argument to the function, and `pdbx_ordinal`,
    which is computed.  The capitalization should match that in the mmCIF dictionary.
    In particular, the following data items are supported:
        'name', 'version', 'location', 'classification',
        'os', 'type', 'citation_id'
    """
    software = _add_software(model, name, info, metadata=metadata)

    # update metadata
    if software is not None:
        software._set_metadata(model)


def _add_software(model, name, info, *, metadata=None, return_existing=False):
    # do bulk of add_software's work, but just return revised tables
    # and don't update metadata
    software, = get_mmcif_tables_from_metadata(model, ['software'], metadata=metadata)

    if software is not None and software.field_has('name', name):
        if return_existing:
            return software
        return None

    # construct new table entries
    possible_items = [
        'name', 'version', 'location', 'classification',
        'os', 'type', 'citation_id', 'pdbx_ordinal'
    ]
    software_items = ['name']
    software_data = [name]
    if software is None:
        max_ordinal = 0
    else:
        ordinals = software.fields(['pdbx_ordinal'])
        max_ordinal = max(0, *(int(x[0]) if x[0].isdigit() else 0 for x in ordinals))
    cinfo = {k.casefold(): (k, v) for k, v in info.items()}
    if 'name' in cinfo:
        del cinfo['name']
    if 'pdbx_ordinal' in cinfo:
        del cinfo['pdbx_ordinal']
    cinfo['pdbx_ordinal'] = ('pdbx_ordinal', str(max_ordinal + 1))
    for i in possible_items:
        ci = i.casefold()
        if ci not in cinfo:
            continue
        software_items.append(i)
        software_data.append(cinfo[ci][1])
        del cinfo[ci]
    for k, v in cinfo.values():
        software_items.append(k)
        software_data.append(v)
    new_software = CIFTable("software", software_items, software_data)

    # combine tables
    if new_software.num_rows() > 0:
        if software is None:
            software = new_software
        else:
            software.extend(new_software)

    # return revised table
    return software


def get_cif_tables(filename, table_names, *, all_data_blocks=False):
    """Supported API. Extract CIF tables from a file

    Parameters
    ----------
    filename : str
        The name of the file.
    table_names : list of str
        A list of CIF category names.
    all_data_blocks : bool
        If true, return tables from data blocks in file.  Default is False.

    Returns
    -------
        list or dictionary
            If all_data_blocks is false, return list of CIF tables found and
            all of the data values in a :py:class:`CIFTable`.
            If all_data_blocks is true, return an ordered dictionary of tables
            per data block.
    """
    from os import path
    if path.exists(filename):
        from . import _mmcif
        all_data = _mmcif.extract_CIF_tables(filename, table_names, all_data_blocks)
    else:
        all_data = []

    def convert_tables(data, table_names):
        tlist = []
        for name in table_names:
            if name not in data:
                tlist.append(CIFTable(name))
            else:
                tags, values = data[name]
                tlist.append(CIFTable(name, tags, values))
        return tlist

    if not all_data_blocks:
        return convert_tables(all_data, table_names)
    result = []
    for block_name, data in all_data:
        tlist = convert_tables(data, table_names)
        result.append((block_name, tlist))
    return result


def get_mmcif_tables_from_metadata(model, table_names, *, metadata=None):
    """Supported API. Extract mmCIF tables from previously read metadata

    Parameters
    ----------
    model : instance of a :py:class:`~chimerax.atomic.AtomicStructure`
        The model.
    table_names : list of str
        A list of mmCIF category names.
    metadata : optional metadata dictonary
        Allow reuse of existing metadata dictionary.
    """
    if metadata is None:
        try:
            metadata = model.metadata
        except AttributeError:
            raise ValueError("Expected a structure")
    tlist = []
    for n in table_names:
        n = n.casefold()
        if n not in metadata or (n + ' data') not in metadata:
            tlist.append(None)
        else:
            info = metadata[n]
            values = metadata[n + ' data']
            tlist.append(CIFTable(info[0], info[1:], values))
    return tlist


class TableMissingFieldsError(ValueError):
    """Supported API. Required field is missing"""
    pass


cdef class CIFTable:
    """
    Supported API. Present a table interface for a (mm)CIF category

    Tags should be in the mixed case version given in the associated dictionary
    """
    cdef str table_name
    cdef list _tags, _folded_tags
    cdef list _data

    def __init__(self, str table_name not None, tags=None, data=None):
        self.table_name = table_name
        self._tags = [] if tags is None else list(tags)
        self._folded_tags = [t.casefold() for t in self._tags]
        self._data = [] if data is None else data
        n = len(self._tags)
        if n == 0:
            assert len(self._data) == 0
        else:
            assert len(self._data) % n == 0

    def __bool__(self):
        """True if not empty"""
        return len(self._tags) != 0 and len(self._data) != 0

    def __eq__(self, CIFTable other):
        if other is None:
            return False
        return self._tags == other._tags and self._data == other._data

    def __repr__(self):
        num_columns = len(self._tags)
        if num_columns == 0:
            num_rows = 0
        else:
            num_rows = len(self._data) / num_columns
        return "CIFTable(%s, %s, ...[%dx%d])" % (
            self.table_name, self._tags, num_rows, num_columns)

    def _set_metadata(self, model, metadata=None):
        from chimerax.atomic.structure import Structure as StructureClass
        assert isinstance(model, StructureClass)
        tag_line = [self.table_name] + self._tags
        if metadata:
            metadata[self.table_name] = tag_line
            metadata[self.table_name + " data"] = self._data
        model.set_metadata_entry(self.table_name, tag_line)
        model.set_metadata_entry(self.table_name + " data", self._data)

    def mapping(self, key_names, value_names, foreach_names=None):
        """Supported API. Return a dictionary for subset of the table

        Parameters
        ----------
        key_names : a field name or list of field names
        value_names : a field name or list of field names
        foreach_names : optional field name or list of field names

        If foreach_names is not given, then the resulting dictionary is whose
        keys are the values of the key field(s) and the values are the values
        of the value field(s).  If foreach_names is given, then the result is
        a dicttionary of dictionaries, with the nested dictionaries as above,
        and the outer dictionary's keys are the values of the foreach fields.
        """
        from itertools import chain
        single_key = isinstance(key_names, str)
        if single_key:
            key_names = [key_names]
        single_value = isinstance(value_names, str)
        if single_value:
            value_names = [value_names]
        single_foreach = isinstance(foreach_names, str)
        if single_foreach:
            foreach_names = [foreach_names]
        elif foreach_names is None:
            foreach_names = []
        t = self._folded_tags
        n = len(t)
        for name in chain(key_names, value_names, foreach_names):
            if name.casefold() not in t:
                from chimerax.core.commands.cli import commas, plural_form
                have = commas(['"%s"' % t for t in self._tags], 'and')
                have_noun = plural_form(self._tags, 'field')
                raise TableMissingFieldsError(
                    'Field "%s" not in table "%s", have %s %s'
                    % (name, self.table_name, have_noun, have))
        key_columns = [self._data[t.index(k.casefold())::n] for k in key_names]
        value_columns = [self._data[t.index(v.casefold())::n] for v in value_names]
        if single_key:
            keys = key_columns[0]
        else:
            keys = zip(*key_columns)
        if single_value:
            values = value_columns[0]
        else:
            values = zip(*value_columns)
        if not foreach_names:
            return dict(zip(keys, values))

        foreach_columns = [self._data[t.index(f.casefold())::n] for f in foreach_names]
        if single_foreach:
            foreachs = foreach_columns[0]
        else:
            foreachs = zip(*foreach_columns)
        m = {}
        for f, k, v in zip(foreachs, keys, values):
            m.set_default(f, {})[k] = v
        return m

    def fields(self, field_names, *, int allow_missing_fields=False, missing_value=''):
        """Supported API. Return subset of rows of the table for the given fields

        Parameters
        ----------
        field_names : a sequence of the field names
        allow_missing_fields : optional boolean (default False)
        missing_value : optional missing_value object (default '')

        A list of tuples is return where each tuple's items are the values
        corresponding to the given field names in the given order.  If
        missing fields are allowed, then the corresponding items are the
        missing_value object.
        """
        cdef int i, n
        t = self._folded_tags
        n = len(self._folded_tags)
        if allow_missing_fields:
            from itertools import zip_longest
            fi = []
            for fn in field_names:
                try:
                    fi.append(t.index(fn.casefold()))
                except ValueError:
                    fi.append(-1)
            ftable = list(zip_longest(
                *(self._data[i::n] if i >= 0 else [] for i in fi),
                fillvalue=missing_value))
        else:
            missing = [fn for fn in field_names if fn.casefold() not in t]
            if missing:
                from chimerax.core.commands.cli import commas, plural_form
                missed = commas(['"%s"' % m for m in missing], 'and')
                missed_noun = plural_form(missing, 'Field')
                missed_verb = plural_form(missing, 'is', 'are')
                have = commas(['"%s"' % t for t in self._tags], 'and')
                have_noun = plural_form(self._tags, 'field')
                raise TableMissingFieldsError('%s %s %s not in table "%s", have %s %s' % (
                    missed_noun, missed, missed_verb, self.table_name, have_noun,
                    have))
            fi = tuple(t.index(fn.casefold()) for fn in field_names)
            ftable = list(zip(*(self._data[i::n] for i in fi)))
        return ftable

    def extend(self, CIFTable table not None):
        """Supported API. Extend mmCIF table

        Parameters
        ----------
        table : CIFTable to add on to current table

        If a column with a given tag exists, then that column is extended.
        Otherwise, a new column is added to the table.
        """
        if self.table_name != table.table_name:
            raise ValueError("incompatible tables")
        num_old_columns = len(self._tags)
        old_columns = [self._data[i::num_old_columns] for i in range(num_old_columns)]
        num_old_rows = len(old_columns[0])
        num_new_columns = len(table._tags)
        new_columns = [table._data[i::num_new_columns] for i in range(num_new_columns)]
        num_new_rows = len(new_columns[0])
        # extend existing columns
        new_unknown = ['?'] * num_new_rows
        new_data = dict(zip(table._folded_tags, new_columns))
        for t, c in zip(self._folded_tags, old_columns):
            d = new_data.pop(t, None)
            if d is None:
                c.extend(new_unknown)
            else:
                c.extend(d)
        # add additional columns if needed
        if new_data:
            old_tags = self._tags[:]
            old_unknown = ['?'] * num_old_rows
            for folded_tag, c in new_data.items():
                for tag in table._tags:
                    if folded_tag == tag.casefold():
                        break
                old_tags.append(tag)
                old_columns.append(old_unknown + c)
            self._tags = old_tags
        from chimerax.core.utils import flattened
        self._data = flattened(zip(*old_columns), return_type=list)

    def has_field(self, field_name):
        """Supported API. Return if given field name is in the table"""
        field_name = field_name.casefold()
        return field_name in self._folded_tags

    def field_has(self, field_name, value):
        """Supported API. Return if given field has the given value"""
        field_name = field_name.casefold()
        try:
            i = self._folded_tags.index(field_name)
        except ValueError:
            return False
        n = len(self._tags)
        return value in self._data[i::n]

    def num_rows(self):
        """Supported API. Return number of rows in table"""
        if len(self._tags) == 0:
            return 0
        return len(self._data) // len(self._tags)

    def print(self, file=None, *, int fixed_width=False):
        """Supported API. Print contents of table to given file

        Parameters
        ----------
        file : CIFTable to add on to current table
        fixed_width : true if fixed width columns should be used

        The fixed width column output matches the PDBx/mmCIF style syntax.
        If fixed_width is asked for and it is not possible to have
        fixed width columns (e.g., there is a newline in a string field),
        then the first row is broken up into multiple lines.
        """
        cdef int n, bad_fixed_width, first, i
        if len(self._data) == 0:
            return
        n = len(self._tags)
        if n == 0:
            return
        assert len(self._data) % n == 0
        if file is None:
            file = sys.stdout
        if n == len(self._data):
            for t, v in zip(self._tags, self._data):
                print(f'_{self.table_name}.{t}', quote(v), file=file)
        else:
            print('loop_', file=file)
            for t in self._tags:
                print(f'_{self.table_name}.{t}', file=file)
            data = [quote(x) for x in self._data]
            if not fixed_width:
                for i in range(0, len(self._data), n):
                    print(' '.join(data[i:i + n]), file=file)
            else:
                bad_fixed_width = False
                columns = [data[i::n] for i in range(n)]
                try:
                    widths = [max(len(f) if f[0] != '\n' else sys.maxsize for f in c) for c in columns]
                except Exception:
                    bad_fixed_width = True
                else:
                    bad_fixed_width = sys.maxsize in widths
                if bad_fixed_width:
                    first = True
                    for i in range(0, len(data), n):
                        if first:
                            first = False
                            print(' '.join(data[i:i + 1]), file=file)
                            print(' '.join(data[i + 1:i + n]), file=file)
                        else:
                            print(' '.join(data[i:i + n]), file=file)
                else:
                    fmt = ''.join(['%%-%ds ' % w for w in widths])
                    for i in range(0, len(data), n):
                        print(fmt % tuple(data[i:i + n]), file=file)
        print('#', file=file)  # PDBx/mmCIF style


# TODO: @deprecated(version='1.1', reason='Use get_cif_tables() instead')
def get_mmcif_tables(filename, table_names):
    """Deprecated API.  Use get_cif_tables() instead."""
    return get_cif_tables(filename, table_names)


def fetch_ccd(session, ccd_id, ignore_cache=False):
    """Get structure for CCD component"""
    from chimerax.atomic.structure import AtomicStructure
    # TODO: support ignore_cache
    ccd_id = ccd_id.upper()  # all current CCD entries are in uppercase
    try:
        ccd = find_template_residue(session, ccd_id)
    except ValueError:
        raise UserError("Unknown CCD ligand name")
    ccd_atoms = ccd.atoms
    ccd_bonds = set()
    for a in ccd_atoms:
        ccd_bonds.update(a.bonds)

    new_structure = AtomicStructure(session, name=ccd_id)
    new_residue = new_structure.new_residue(ccd_id, 'A', 1)
    new_atoms = {}
    for a in ccd_atoms:
        new_atom = new_structure.new_atom(a.name, a.element)
        new_atom.coord = a.coord
        new_atoms[a] = new_atom
        new_residue.add_atom(new_atom)

    for b in ccd_bonds:
        atoms = b.atoms
        new_a0 = new_atoms[atoms[0]]
        new_a1 = new_atoms[atoms[1]]
        new_structure.new_bond(new_a0, new_a1)

    from chimerax.pdb import process_chem_name
    new_structure.html_title = process_chem_name(ccd.description)

    return [new_structure], f"Opened CCD {ccd_id}"


def non_standard_bonds(bonds, selected_only=False, displayed_only=False):
    from . import _mmcif
    from chimerax.atomic import Bonds
    disulfide, covalent = _mmcif.non_standard_bonds(bonds, selected_only, displayed_only)
    if disulfide is not None:
        disulfide = Bonds(disulfide)
    if covalent is not None:
        covalent = Bonds(covalent)
    return disulfide, covalent
