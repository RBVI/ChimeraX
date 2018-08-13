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

def open_pdb(session, stream, file_name, *, auto_style=True, coordsets=False, atomic=True,
             max_models=None, log_info=True):

    path = stream.name if hasattr(stream, 'name') else None

    from . import _pdbio
    pointers = _pdbio.read_pdb_file(stream, log=session.logger, explode=not coordsets, atomic=atomic)
    stream.close()

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
                from chimerax.std_commands.coordset import coordset_slider
                coordset_slider(session, mc)
    if models:
        m = models[0]
        title_recs = m.metadata.get('TITLE', None)
        if title_recs:
            text = collate_records_text(title_recs)
            m.html_title = process_chem_name(text.strip(), sentences=True)
            m.has_formatted_metadata = lambda ses: True
            m.get_formatted_metadata = lambda ses, *, m=m, verbose=False, **kw:\
                _get_formatted_metadata(m, ses, verbose)

    return models, info


def save_pdb(session, path, *, models=None, selected_only=False, displayed_only=False,
        all_coordsets=False, pqr=False, rel_model=None, serial_numbering="h36"):
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

    from . import _pdbio
    file_per_model = "[NAME]" in path or "[ID]" in path
    if file_per_model:
        for m, xform in zip(models, xforms):
            file_name = path.replace("[ID]", m.id_string()).replace("[NAME]", m.name)
            _pdbio.write_pdb_file([m.cpp_pointer], file_name, selected_only=selected_only,
                displayed_only=displayed_only, xforms=[xform],
                all_coordsets=all_coordsets, pqr=pqr, h36=(serial_numbering == "h36"))
    else:
        _pdbio.write_pdb_file([m.cpp_pointer for m in models], path, selected_only=selected_only,
            displayed_only=displayed_only, xforms=xforms, all_coordsets=all_coordsets, pqr=pqr,
            h36=(serial_numbering == "h36"))


_pdb_sources = {
#    "rcsb": "http://www.pdb.org/pdb/files/%s.pdb",
    "rcsb": "http://files.rcsb.org/download/%s.pdb",
    "pdbe": "http://www.ebi.ac.uk/pdbe/entry-files/download/pdb%s.ent",
    # "pdbj": "https://pdbj.org/rest/downloadPDBfile?format=pdb&id=%s",
}


def fetch_pdb(session, pdb_id, *, fetch_source="rcsb", ignore_cache=False, **kw):
    if len(pdb_id) != 4:
        from chimerax.core.errors import UserError
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
        from chimerax.core.fetch import fetch_file
        filename = fetch_file(session, url, 'PDB %s' % pdb_id, pdb_name, 'PDB',
                              ignore_cache=ignore_cache)

    session.logger.status("Opening PDB %s" % (pdb_id,))
    from chimerax.core import io
    models, status = io.open_data(session, filename, format='pdb', name=pdb_id, **kw)
    return models, status



def fetch_pdb_pdbe(session, pdb_id, **kw):
    return fetch_pdb(session, pdb_id, fetch_source="pdbe", **kw)


def fetch_pdb_pdbj(session, pdb_id, **kw):
    return fetch_pdb(session, pdb_id, fetch_source="pdbj", **kw)


def register_pdb_format():
    from chimerax.core import io
    from chimerax.atomic import structure
    io.register_format(
        "PDB", structure.CATEGORY, (".pdb", ".pdb1", ".ent", ".pqr"), ("pdb",),
        mime=("chemical/x-pdb", "chemical/x-spdbv"),
        reference="http://wwpdb.org/docs.html#format",
        open_func=open_pdb, export_func=save_pdb)
    from chimerax.core.commands import BoolArg, ModelsArg, ModelArg, IntArg, EnumOf
    from chimerax.core.commands.cli import add_keyword_arguments
    add_keyword_arguments('open', {'coordsets':BoolArg, 'auto_style':BoolArg,
                                   'atomic': BoolArg, 'max_models':IntArg, 'log_info':BoolArg})
    add_keyword_arguments('save', {'models':ModelsArg, 'selected_only':BoolArg,
        'displayed_only':BoolArg, 'all_coordsets':BoolArg, 'pqr':BoolArg,
        'rel_model':ModelArg, 'serial_numbering': EnumOf(("amber", "h36"))})


def register_pdb_fetch():
    from chimerax.core import fetch
    fetch.register_fetch('pdb', fetch_pdb, 'pdb', prefixes = [])
    fetch.register_fetch('pdbe', fetch_pdb_pdbe, 'pdb', prefixes = [])
    # PDBj is unreliable for PDB format, mmCIF seemed okay - CH 2dec16
    # fetch.register_fetch('pdbj', fetch_pdb_pdbj, 'pdb', prefixes = [])


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

def _get_formatted_metadata(model, session, verbose):
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

    """
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
    """
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

