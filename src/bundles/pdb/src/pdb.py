# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
pdb: PDB format support
=======================

Read Protein DataBank (PDB) files.
"""

def open_pdb(session, stream, file_name=None, *, auto_style=True, coordsets=False, atomic=True,
             max_models=None, log_info=True, combine_sym_atoms=True, segid_chains=False,
             slider=True, missing_coordsets="renumber"):
    """Read PDB data from a file or stream and return a list of models and status information.

    ``stream`` is either a string a string with a file system path to a PDB file, or an open input
    stream to PDB data.

    ``file_name`` is the name to give to the resulting model(s).  Typically only needed if the input
    is an anonymous stream or the input file name wouldn't be a good model name.

    ``auto_style`` is passed through to the :py:class:`~chimerax.atomic.structure.Structure`
    or :py:class:`~chimerax.atomic.structure.AtomicStructure` constructor.

    ``coordsets`` controls whether a multi-MODEL PDB is opened as a list of structures or as
    a single structure with multiple coordinate sets.

    ``atomic`` controls whether AtomicStructure or Structure is used as the class for the structure.
    The latter should be used for PDB files that don't actually contain atomic data per se, like SAX
    "PDB" files or coarse-grain models.

    ``max_models`` limits the number of models this routine can return.

    ``log_info`` is passed through to the :py:class:`~chimerax.atomic.structure.Structure`
    or :py:class:`~chimerax.atomic.structure.AtomicStructure` constructor.

    ``combine_sym_atoms`` controls whether otherwise identical atoms with no bonds that are also very
    close together in space should be combined into a single atom.

    ``segid_chains`` controls whether the chain ID should come from the normal chain ID columns or from
    the "segment ID" columns.

    ``slider`` controls whether a slider tool is shown when a multi-model PDB file is opened as a
    trajectory.

    ``missing_coordsets`` is for the rare case where MODELs are being collated into a trajectory and the
    MODEL numbers are not consecutive.  The possible values are 'fill' (fill in the missing with copies
    of the preceding coord set), 'ignore' (don't fill in; use MODEL number as is for coordset ID), and
    'renumber' (don't fill in and use the next available coordset ID).
    """

    from chimerax.core.errors import UserError
    if isinstance(stream, str):
        path = stream
        stream = open(stream, 'r')
    else:
        path = stream.name if hasattr(stream, 'name') else None

    if file_name is None:
        if path:
            from os.path import basename
            file_name = basename(path)
        else:
            file_name = 'structure'
            
    from . import _pdbio
    try:
        pointers = _pdbio.read_pdb_file(stream, session.logger, not coordsets, atomic, segid_chains,
            ['fill', 'ignore', 'renumber'].index(missing_coordsets))
    except ValueError as e:
        if 'non-ASCII' in str(e):
            raise UserError(str(e))
        raise
    finally:
        stream.close()

    if atomic:
        from chimerax.atomic.structure import AtomicStructure as StructureClass
    else:
        from chimerax.atomic.structure import Structure as StructureClass
    models = [StructureClass(session, name=file_name, c_pointer=p, auto_style=auto_style, log_info=log_info)
        for p in pointers]
    from numpy import isnan
    for m in models:
        if isnan(m.atoms.coords).any():
            for dm in models:
                dm.delete()
            raise UserError("Some X/Y/Z coordinate values in the '%s' PDB file are not numbers" % file_name)

    if max_models is not None:
        for m in models[max_models:]:
            m.delete()
        models = models[:max_models]

    if path:
        for m in models:
            m.filename = path
    if combine_sym_atoms:
        for m in models:
            m.combine_sym_atoms()

    info = ''
    if coordsets:
        num_cs = 0
        for m in models:
            num_cs += m.num_coordsets
        info = '%s has %d coordinate sets' % (file_name, num_cs)
        if slider and session.ui.is_gui:
            mc = [m for m in models if m.num_coordsets > 1]
            if mc:
                from chimerax.std_commands.coordset import coordset_slider
                coordset_slider(session, mc)
    if models:
        m = models[0]
        set_logging_info(m)

    return models, info


def save_pdb(session, output, *, models=None, selected_only=False, displayed_only=False,
        all_coordsets=False, pqr=False, rel_model=None, serial_numbering="h36",
        polymeric_res_names = None):
    """Write PDB data to a file.

    ``output`` is a file system path to a writable location.  It can contain the strings "[NAME]"
    and/or "[ID]", which will be replaced with the model's name/id, respectively.

    ``models`` is a list of models to output.  If not specified, all structure models will be output.

    ``selected_only`` controls whether only currently selected atoms should be output.

    ``displayed_only`` controls whether only currently displayed atoms should be output.

    ``all_coordsets`` controls whether or not, for a multi-coordset model, all coordsets should be
    written out (using MODEL records) or just the current coordinate set.

    ``pqr`` controls whether ATOM/HETATM records will be written out in non-standard PQR format or not.

    ``rel_model`` if given, is a model that the output coordinates should be written "relative to", i.e.
    whatever then inverse of ``relmodel``'s current transformation is, apply that to the atomic coordinates
    before outputting them.

    ``serial_numbering`` controls how serial numbers are output when they would exceed PDB column limits.
    "h36" means to use Hybrid-36 numbering.  "amber" means to steal a column from the "ATOM   " record and
    not correct them in other types of records (e.g. CONECT).

    ``polymeric_res_names`` is a list of residue names that should be considered "standard" as far as the
    the output of ATOM vs. HETATM goes.  If not specified, the residue names that the RCSB considers
    standard will be used.
    """
    from chimerax.core.errors import UserError
    if models is None:
        models = session.models
    from chimerax.atomic import Structure
    models = [m for m in models if isinstance(m, Structure)]
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

    # If model came from mmCIF, try to generate CRYST1 record if we can
    for s in models:
        if not hasattr(s, "metadata") or 'CRYST1' in s.metadata:
            continue
        from chimerax.mmcif import TableMissingFieldsError, get_mmcif_tables_from_metadata as get_tables
        tables = get_tables(s, ['cell', 'symmetry'])
        if None in tables:
            continue
        try:
            l_a, l_b, l_c, a_a, a_b, a_g, z, h_m = tables[0].fields(['length_a', 'length_b', 'length_c',
                'angle_alpha', 'angle_beta', 'angle_gamma', 'Z_PDB'])[0] + tables[1].fields(
                ['space_group_name_H-M'])[0]
        except TableMissingFieldsError:
            continue
        if z.strip() == '?':
            z = ""
        s.set_metadata_entry('CRYST1',
            ["CRYST1%9s%9s%9s%7s%7s%7s %-11s%4s" % (l_a, l_b, l_c, a_a, a_b, a_g, h_m, z)])
    from . import _pdbio
    if polymeric_res_names is None:
        polymeric_res_names = _pdbio.standard_polymeric_res_names
    file_per_model = "[NAME]" in output or "[ID]" in output
    if file_per_model:
        for m, xform in zip(models, xforms):
            file_name = output.replace("[ID]", m.id_string).replace("[NAME]", m.name)
            _pdbio.write_pdb_file([m.cpp_pointer], file_name, selected_only,
                displayed_only, [xform], all_coordsets,
                pqr, (serial_numbering == "h36"), polymeric_res_names, session.logger)
    else:
        _pdbio.write_pdb_file([m.cpp_pointer for m in models], output, selected_only,
            displayed_only, xforms, all_coordsets, pqr,
            (serial_numbering == "h36"), polymeric_res_names, session.logger)

_pdb_sources = {
#    "rcsb": "http://www.pdb.org/pdb/files/%s.pdb",
    "rcsb": "http://files.rcsb.org/download/%s.pdb",
    "pdbe": "http://www.ebi.ac.uk/pdbe/entry-files/download/pdb%s.ent",
    "pdbj": "https://pdbj.org/rest/downloadPDBfile?format=pdb&id=%s",
}

def fetch_pdb(session, pdb_id, *, fetch_source="rcsb", ignore_cache=False,
        structure_factors=False, over_sampling=1.5, # for ChimeraX-Clipper plugin
        **kw):
    from chimerax.core.errors import UserError
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
        from chimerax.core.fetch import fetch_file
        filename = fetch_file(session, url, 'PDB %s' % pdb_id, pdb_name, 'PDB',
                              ignore_cache=ignore_cache)

    session.logger.status("Opening PDB %s" % (pdb_id,))
    models, status = session.open_command.open_data(filename, format='pdb',
        name=pdb_id, **kw)
    if structure_factors:
        sf_file = fetch_cif.fetch_structure_factors(session, pdb_id,
            fetch_source=fetch_source, ignore_cache=ignore_cache)
        from chimerax.clipper import get_map_mgr
        mmgr = get_map_mgr(models[0], create=True)
        if over_sampling < 1:
            warn_str = ('Map over-sampling rate cannot be less than 1. Resetting to 1.0')
            session.logger.warning(warn_str)
            over_sampling = 1
        mmgr.add_xmapset_from_file(sf_file, oversampling_rate = over_sampling)
        return [mmgr.crystal_mgr], status

    return models, status

def fetch_pdb_pdbe(session, pdb_id, **kw):
    return fetch_pdb(session, pdb_id, fetch_source="pdbe", **kw)

def fetch_pdb_pdbj(session, pdb_id, **kw):
    return fetch_pdb(session, pdb_id, fetch_source="pdbj", **kw)

def collate_records_text(records, multiple_results=False):
    if multiple_results:
        collation = []
        cur_record_set = []
        for record in records:
            if record[8:11].strip() == "":
                if cur_record_set:
                    collation.append(collate_records_text(cur_record_set))
                cur_record_set = [record]
            else:
                cur_record_set.append(record)
        if cur_record_set:
            collation.append(collate_records_text(cur_record_set))
        return collation
    text = ""
    for record in records:
        text += " " + record[10:].strip()
    return text

def collate_het_records(records):
    return collate_subtyped_records(records, 8, 10, 11, 14, 15)

def collate_jrnl_records(records):
    return collate_subtyped_records(records, 16, 18, 12, 16, 19)

def collate_subtyped_records(records, cont_start, cont_end, type_start, type_end, data_start):
    collated = {}
    for rec in records:
        if rec[cont_start:cont_end].strip():
            # continuation
            collated[subtype] = collated[subtype] + rec[data_start:].strip()
        else:
            subtype = rec[type_start:type_end].strip()
            collated[subtype] = rec[data_start:].strip()
    return collated

def _get_formatted_res_info(model, *, standalone=True):
    def update_nonstd(model, nonstd_info):
        names = collate_het_records(model.metadata.get('HETNAM', {}))
        syns = collate_het_records(model.metadata.get('HETSYN', {}))
        for het, info in list(nonstd_info.items()):
            if het not in names and het not in syns:
                continue
            name = process_chem_name(names[het]) if het in names else info[1]
            syn = process_chem_name(syns[het]) if het in syns else info[2]
            nonstd_info[het] = (info[0], name, syn)
    return format_nonstd_res_info(model, update_nonstd, standalone)

# also used by mmcif
def format_nonstd_res_info(model, update_nonstd_res_info, standalone):
    from chimerax.pdb import process_chem_name
    html = ""
    nonstd_res_names = model.nonstandard_residue_names
    if nonstd_res_names:
        nonstd_info = { rn:(rn, "(%s)" % rn, None) for rn in nonstd_res_names }
        update_nonstd_res_info(model, nonstd_info)
        def fmt_component(abbr, name, syns):
            if abbr.isdigit():
                selector = "::name=&quot;%s&quot;" % abbr
            else:
                selector = ":%s" % abbr
            text = '<a title="select residue" href="cxcmd:sel %s">%s</a> &mdash; ' % (selector, abbr)
            if name:
                text += '<a title="show residue info" href="http://www.rcsb.org/ligand/%s">%s</a>' % (abbr,
                    process_chem_name(name))
                if syns:
                    text += " (%s)" % process_chem_name(syns)
            elif syns:
                text += process_chem_name(syns)
            else:
                text += "(no full residue name provided)"
            return text
        if standalone:
            from chimerax.core.logger import html_table_params
            html = "<table %s>\n" % html_table_params
            html += ' <thead>\n'
            html += '  <tr>\n'
            html += '   <th>Non-standard residues in %s</th>\n' % model
            html += '  </tr>\n'
            html += ' </thead>\n'
            html += ' <tbody>\n'

        for i, info in enumerate(nonstd_info.values()):
            abbr, name, synonyms = info
            html += '  <tr>\n'
            formatted = fmt_component(abbr, name, synonyms)
            if i == 0 and not standalone:
                if len(nonstd_info) > 1:
                    html += '   <th rowspan="%d">Non-standard residues</th>\n' % len(nonstd_info)
                else:
                    html += '   <th>Non-standard residue</th>\n'
            html += '   <td>%s</td>\n' % formatted
            html += '  </tr>\n'

        if standalone:
            html += ' </tbody>\n'
            html += "</table>"
    return html

# also used by mmcif
def format_source_name(common_name, scientific_name, genus, species, ncbi_id):
    text = ""
    if scientific_name:
        text = scientific_name
    else:
        if genus:
            text = genus if not species else genus + " " + species
        else:
            text = species

    if text and ncbi_id:
        text = process_chem_name(text, sentences=True)
        text = '<a href="https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=' \
            + ncbi_id + '">' + text + '</a>'
    if common_name:
        if text:
            common_name = process_chem_name(common_name.lower())
            if not ncbi_id:
                from html import escape
                text = escape(text)
            text = text + ' (%s)' % common_name
        else:
            common_name = process_chem_name(common_name.lower(), sentences=True)
            if ncbi_id:
                text = '<a href="https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=' \
                    + ncbi_id + '">' + common_name + '</a>'
            else:
                text = common_name
    return text

def process_chem_name(name, use_greek=True, probable_abbrs=False, sentences=False):
    if name.isupper() and (" " in name or len(name) > 5):
        # probable non-abbreviation all uppercase:  need to downcase as appropriate
        text = ""
        word = ""
        capitalize = sentences
        for c in name.strip().lower():
            if capitalize and not c.isspace():
                c = c.upper()
                capitalize = False
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
            if sentences and text[-2:] == ". ":
                capitalize = True
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
    from html import escape
    return escape(text)

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
    from chimerax.atomic import Element
    if len(word) == 2 and word.capitalize() in Element.names:
        return word.capitalize()
    if word == "ph":
        return "pH"
    if set(list(word)) <= set(["i", "v", "x"]):
        # Roman numeral
        return word.upper()
    if use_greek and word in greek_letters:
        return greek_letters[word]
    if probable_abbrs and len(word) < 5 and word != "cell":
        return word.upper()
    if not probable_abbrs and len(word) < 5:
        for i, c in enumerate(word):
            if c in "aeiouy":
                if len(word) < 4 or (i > 0 and i < 3):
                    break
        else:
            return word.upper()
    segs = []
    for seg in word.split('-'):
        if use_greek and seg in greek_letters:
            segs.append(greek_letters[seg])
        elif seg.startswith("rna") or seg.startswith("dna") or seg.startswith("nmr"):
            segs.append(seg[:3].upper() + seg[3:])
        elif seg.endswith("rna") or seg.endswith("dna") or seg.endswith("nmr"):
            segs.append(seg[:-3] + seg[-3:].upper())
        else:
            segs.append(word)
    return '-'.join(segs)

def set_logging_info(m):
    # also used by Chimera->ChimeraX exporter
    title_recs = m.metadata.get('TITLE', None)
    if title_recs:
        text = collate_records_text(title_recs)
        m.html_title = process_chem_name(text.strip(), sentences=True)
        m.has_formatted_metadata = lambda ses: True
        from types import MethodType
        from weakref import proxy
        m.get_formatted_metadata = MethodType(_get_formatted_metadata, proxy(m))
        m.get_formatted_res_info = MethodType(_get_formatted_res_info, proxy(m))

def _get_formatted_metadata(model, session, *, verbose=False):
    from chimerax.core.logger import html_table_params
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

    # citation
    cite = collate_jrnl_records(model.metadata.get('JRNL', []))
    if 'TITL' in cite:
        cite_text = process_chem_name(cite['TITL'], sentences=True)
        if 'DOI' in cite:
            cite_text = '<a href="http://dx.doi.org/%s">%s</a>' % (cite['DOI'], cite_text)
        if 'PMID' in cite:
            cite_text += ' PMID: <a href="http://www.ncbi.nlm.nih.gov/pubmed/%s">%s</a>' % (
                cite['PMID'], cite['PMID'])
        html += '  <tr>\n'
        html += '   <th>Citation</th>\n'
        html += '   <td>%s</td>\n' % cite_text
        html += '  </tr>\n'

    # non-standard residues
    html += model.get_formatted_res_info(standalone=False)

    # source
    engineered = None
    genes = set()
    hosts = set()
    source_recs = model.metadata.get("SOURCE", None)
    if source_recs:
        source_info = collate_records_text(source_recs)
        gene = host = None
        for info in source_info.split(';'):
            info = info.strip()
            if info.startswith('MOL_ID:'):
                if gene and gene[:2] != (None, None):
                    genes.add(gene)
                if host and host[0] != None:
                    hosts.add(host)
                gene = host = None
                continue
            if info.startswith('ORGANISM_SCIENTIFIC:'):
                sci_org = info[20:].strip()
                if gene:
                    gene = (sci_org, gene[1], gene[2])
                else:
                    gene = (sci_org, None, None)
            if info.startswith('ORGANISM_COMMON:'):
                com_org = info[16:].strip()
                if gene:
                    gene = (gene[0], com_org, gene[2])
                else:
                    gene = (None, com_org, None)
            if info.startswith('ORGANISM_TAXID:'):
                tax_id = info[15:].strip()
                if gene:
                    gene = (gene[0], gene[1], tax_id)
                else:
                    gene = (None, None, tax_id)
            if info.startswith('EXPRESSION_SYSTEM:'):
                expr_sys = info[18:].strip()
                engineered = True
                if host:
                    host = (expr_sys, host[1])
                else:
                    host = (expr_sys, None)
            if info.startswith('EXPRESSION_SYSTEM_TAXID:'):
                taxid = info[24:].strip()
                engineered = True
                if host:
                    host = (host[0], taxid)
                else:
                    host = (None, taxid)
        if gene and gene[:2] != (None, None):
            genes.add(gene)
        if host and host[0] != None:
            hosts.add(host)
        if genes or hosts:
            if engineered is None:
                compound_recs = model.metadata.get("COMPND", None)
                if compound_recs:
                    compound_info = collate_records_text(compound_recs)
                    if 'ENGINEERED:' in compound_info:
                        engineered = 'ENGINEERED: YES' in compound_info
            if engineered is None:
                engineered = False
            if engineered:
                if genes:
                    html += _process_src(genes, "Gene source%s")
                if hosts and verbose:
                    html += _process_src(hosts, "Host organism%s")
            elif genes:
                html += _process_src(genes, "Source%s (natural)")

    """
    # source
    nat, gen = get_mmcif_tables_from_metadata(model, ["entity_src_nat", "entity_src_gen"])
    if nat:
        html += _process_src(nat, "Source%s (natural)", ['common_name', 'pdbx_organism_scientific',
            'genus', 'species', 'pdbx_ncbi_taxonomy_id'])
    if gen:
        html += _process_src(gen, "Gene source%s", ['gene_src_common_name',
            'pdbx_gene_src_scientific_name', 'gene_src_genus', 'gene_src_species',
            'pdbx_gene_src_ncbi_taxonomy_id'])
        html += _process_src(gen, "Host organism%s", ['host_org_common_name',
            'pdbx_host_org_scientific_name', 'host_org_genus', 'host_org_species',
            'pdbx_host_org_ncbi_taxonomy_id'])
    """

    remark_recs = model.metadata.get("REMARK", None)

    # experimental method; resolution
    method = None
    expdta_recs = model.metadata.get("EXPDTA", None)
    if expdta_recs:
        method = collate_records_text(expdta_recs)
    if not method:
        if remark_recs:
            for record in remark_recs:
                if record[7:10] == "200" and "EXPERIMENT TYPE" in record and ":" in record:
                    method = record.split(":")[-1].strip()
                    break
    if method:
        html += '  <tr>\n'
        html += '   <th>Experimental method</th>\n'
        html += '   <td>%s</td>\n' % process_chem_name(method, sentences=True)
        html += '  </tr>\n'
    res = None
    if remark_recs:
        for record in remark_recs:
            if record[7:10] == "  2" and "RESOLUTION" in record and "ANGSTROMS" in record:
                res = ""
                for c in record[11:]:
                    if res:
                        if c.isdigit() or c == '.':
                            res = res + c
                            continue
                        break
                    if c.isdigit():
                        res = c
    if res is not None:
        html += '  <tr>\n'
        html += '   <th>Resolution</th>\n'
        html += '   <td>%s\N{ANGSTROM SIGN}</td>\n' % res
        html += '  </tr>\n'

    html += ' </tbody>\n'
    html += "</table>"

    return html

def _process_src(src, caption):
    rows = list(src)
    html = '  <tr>\n'
    if len(rows) > 1:
        html += '   <th rowspan="%d">%s</th>\n' % (len(rows), caption % 's')
    else:
        html += '   <th>%s</th>\n' % caption % ''
    for i, row in enumerate(rows):
        try:
            scientific_name, common_name, tax_id = rows[0]
        except ValueError:
            scientific_name, tax_id = rows[0]
            common_name = None
        formatted = format_source_name(common_name, scientific_name, None, None, tax_id)
        if i == 0:
            html += '   <td>%s</td>\n' % formatted
            html += '  </tr>\n'
        else:
            html += '  <tr>\n'
            html += '   <td>%s</td>\n' % formatted
            html += '  </tr>\n'
    return html
