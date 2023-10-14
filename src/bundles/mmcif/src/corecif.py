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
corecif: Small Molecule CIF (core_std.dic) format support
=========================================================

Read coreCIF files.
"""

from chimerax.core.errors import UserError

_additional_corecif_categories = (
    "chemical",
    "chemical_formula",
    "citation",
    "citation_author",
    "citation_editor",
    "exptl",
    "journal",
    "publ",
    "publ_author",
    "refine",
    "reflns",
    # Unable to unambigiously determine the category in CIF v1 files, so
    # register categories that are suffixes of previously listed categories.
    "chemical_conn_atom",
    "chemical_conn_bond",
    "exptl_crystal",
    "exptl_crystal_face",
    "journal_index",
    "publ_body",
    "publ_manuscript_incl",
    "reflns_class",
    "reflns_scale",
    "reflns_shell",
)


def open_corecif(session, path, file_name=None, auto_style=True, log_info=True, extra_categories=()):
    # coreCIF parsing requires an uncompressed file

    from . import _mmcif
    categories = _additional_corecif_categories + tuple(extra_categories)
    log = session.logger if log_info else None
    try:
        pointers = _mmcif.parse_coreCIF_file(path, categories, log)
    except _mmcif.error as e:
        raise UserError('mmCIF parsing error: %s' % e)

    if file_name is None:
        from os.path import basename
        file_name = basename(path)
    from chimerax.atomic.structure import AtomicStructure as StructureClass
    models = [StructureClass(session, name=file_name, c_pointer=p, auto_style=auto_style, log_info=log_info)
              for p in pointers]

    for m in models:
        m.filename = path

    info = ''
    for model in models:
        model.is_corecif = True	 # Indicates metadata is from corecif.
        model.is_mmcif = True	 # Indicates metadata is from mmcif.
        _, title = chemical_name(model)
        if not title:
            title = chemical_formula(model)
        if title:
            from chimerax.pdb import process_chem_name
            model.html_title = process_chem_name(title, sentences=True)
        model.has_formatted_metadata = lambda ses: True
        # use proxy to avoid circular ref
        from weakref import proxy
        from types import MethodType
        model.get_formatted_metadata = MethodType(_get_formatted_metadata, proxy(model))
    if log is not None and not models:
        log.warning("No small molecule CIF models found.  Perhaps this is a mmCIF file?\n")
    return models, info


def chemical_name(model, metadata=None):
    from . import mmcif
    chem = mmcif.get_mmcif_tables_from_metadata(model, ["chemical"], metadata=metadata)[0]
    if chem:
        fields = ['name_common', 'name_mineral', 'name_sructure_type', 'name_systematic']
        titles = chem.fields(fields, allow_missing_fields=True)[0]
        for field, title in zip(fields, titles):
            if title:
                return field, title
    return None, None


def chemical_formula(model, metadata=None):
    from . import mmcif
    chem = mmcif.get_mmcif_tables_from_metadata(model, ["chemical_formula"], metadata=metadata)[0]
    if not chem:
        return
    try:
        return chem.fields(['sum'])[0][0]
    except mmcif.TableMissingFieldsError:
        return


def _get_formatted_metadata(model, session, *, verbose=False):
    from html import escape
    from .mmcif import citations, experimental_method, resolution
    from chimerax.core.logger import html_table_params
    from chimerax.pdb import process_chem_name
    html = "<table %s>\n" % html_table_params
    html += ' <thead>\n'
    html += '  <tr>\n'
    html += '   <th colspan="2">Metadata for %s</th>\n' % escape(str(model))
    html += '  </tr>\n'
    html += ' </thead>\n'
    html += ' <tbody>\n'

    metadata = model.metadata  # get once from C++ layer

    kind, name = chemical_name(model)
    if name:
        title = kind.removeprefix("name_").replace('_', ' ').title()
        html += (
            '  <tr>\n'
            f'   <th>{escape(title)} Name</th>\n'
            f'   <td>{escape(name)}</td>\n'
            '  </tr>\n'
        )

    formula = chemical_formula(model, metadata=metadata)
    if formula:
        html += '  <tr>\n'
        html += '   <th>Chemical Formula</th>\n'
        html += '   <td>%s</td>\n' % escape(formula)
        html += '  </tr>\n'

    # citations
    cites = citations(model, metadata=metadata)
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

    # experimental method; resolution
    method = experimental_method(model, metadata=metadata)
    if method:
        html += '  <tr>\n'
        html += '   <th>Experimental method</th>\n'
        html += '   <td>%s</td>\n' % process_chem_name(method, sentences=True)
        html += '  </tr>\n'
    res = resolution(model, metadata=metadata)
    if res is not None:
        html += '  <tr>\n'
        html += '   <th>Resolution</th>\n'
        html += '   <td>%s\N{ANGSTROM SIGN}</td>\n' % escape(res)
        html += '  </tr>\n'

    html += ' </tbody>\n'
    html += "</table>"

    return html


def is_int(name):
    try:
        int(name)
    except ValueError:
        return False
    return True


_corecif_sources = {
        "cod": "https://www.crystallography.net/cif/{id}.cif",
        # since we're using .format(), we can't use id[0:3] below
        "pcod": "https://www.crystallography.net/pcod/cif/{id[0]}/{id[0]}{id[1]}{id[2]}/{id}.cif",
}


def fetch_cod(session, cod_id, fetch_source="cod", ignore_cache=False, **kw):
    """Get coreCIF file by COD identifier via the Internet"""

    if not is_int(cod_id) or len(cod_id) != 7:
        raise UserError('COD identifiers are 7 digits long, got "%s"' % cod_id)

    import os
    cache = fetch_source
    base_url = _corecif_sources.get(fetch_source, None)
    if base_url is None:
        raise UserError('unrecognized coreCIF source "%s"' % fetch_source)
    url = base_url.format(id=cod_id)
    cod_name = f"{cod_id}.cif"
    from chimerax.core.fetch import fetch_file
    filename = fetch_file(session, url, f'{fetch_source.upper()} {cod_id}', cod_name,
                          cache, ignore_cache=ignore_cache)
    # double check that a CIF file was downloaded instead of an
    # HTML error message saying the ID does not exist
    with open(filename, 'r') as f:
        line = f.readline()
        if not line.startswith(('data_', '#')):
            f.close()
            os.remove(filename)
            raise UserError(f"Invalid {fetch_source.upper()} identifier")

    session.logger.status(f"Opening {fetch_source.upper()} {cod_id}")
    models, status = session.open_command.open_data(
        filename, format='corecif', name=cod_id, **kw)
    return models, status


def fetch_pcod(session, cod_id, **kw):
    return fetch_cod(session, cod_id, fetch_source="pcod", **kw)


# CIF markup conventions from:
# https://www.iucr.org/resources/cif/spec/version1.1/semantics#markup
markup = {
    r'\A': '\N{greek capital letter alpha}',
    r'\B': '\N{greek capital letter beta}',
    r'\C': '\N{greek capital letter chi}',
    r'\D': '\N{greek capital letter delta}',
    r'\E': '\N{greek capital letter epsilon}',
    r'\F': '\N{greek capital letter phi}',
    r'\G': '\N{greek capital letter gamma}',
    r'\H': '\N{greek capital letter eta}',
    r'\I': '\N{greek capital letter iota}',
    r'\K': '\N{greek capital letter kappa}',
    r'\L': '\N{greek capital letter lamda}',
    r'\M': '\N{greek capital letter mu}',
    r'\N': '\N{greek capital letter nu}',
    r'\O': '\N{greek capital letter omicron}',
    r'\P': '\N{greek capital letter pi}',
    r'\Q': '\N{greek capital letter theta}',
    r'\R': '\N{greek capital letter rho}',
    r'\S': '\N{greek capital letter sigma}',
    r'\T': '\N{greek capital letter tau}',
    r'\U': '\N{greek capital letter upsilon}',
    r'\W': '\N{greek capital letter omega}',
    r'\X': '\N{greek capital letter xi}',
    r'\Y': '\N{greek capital letter psi}',
    r'\Z': '\N{greek capital letter zeta}',

    r'\a': '\N{greek small letter alpha}',
    r'\b': '\N{greek small letter beta}',
    r'\c': '\N{greek small letter chi}',
    r'\d': '\N{greek small letter delta}',
    r'\e': '\N{greek small letter epsilon}',
    r'\f': '\N{greek small letter phi}',
    r'\g': '\N{greek small letter gamma}',
    r'\h': '\N{greek small letter eta}',
    r'\i': '\N{greek small letter iota}',
    r'\k': '\N{greek small letter kappa}',
    r'\l': '\N{greek small letter lamda}',
    r'\m': '\N{greek small letter mu}',
    r'\n': '\N{greek small letter nu}',
    r'\o': '\N{greek small letter omicron}',
    r'\p': '\N{greek small letter pi}',
    r'\q': '\N{greek small letter theta}',
    r'\r': '\N{greek small letter rho}',
    r'\s': '\N{greek small letter sigma}',
    r'\t': '\N{greek small letter tau}',
    r'\u': '\N{greek small letter upsilon}',
    r'\w': '\N{greek small letter omega}',
    r'\x': '\N{greek small letter xi}',
    r'\y': '\N{greek small letter psi}',
    r'\z': '\N{greek small letter zeta}',

    r"\'": '\N{combining acute accent}',
    r'\`': '\N{combining grave accent}',
    r'\^': '\N{combining circumflex accent}',
    r'\,': '\N{combining cedilla}',
    r'\"': '\N{combining diaeresis}',
    r'\~': '\N{combining tilde}',
    r'\;': '\N{combining ogonek}',
    r'\>': '\N{combining double acute accent}',
    r'\=': '\N{combining macron}',
    r'\.': '\N{combining dot above}',
    r'\<': '\N{combining caron}',
    r'\(': '\N{combining breve}',

    r'\%a': '\N{latin small letter a with ring above}',
    r'\/o': '\N{latin small letter o with stroke}',
    r'\?i': '\N{latin small letter dotless i}',
    r'\/l': '\N{latin small letter l with stroke}',
    r'\&s': '\N{latin small letter sharp s}',
    r'\/d': '\N{latin small letter d with stroke}',

    r'\%': '\N{degree sign}',
    r'--': '\N{em dash}',
    r'---': '\N{figure dash}',  # single bond, must be followed by space
    r'\\db': '\N{figure dash}',  # double bond, must be followed by space
    r'\\tb': '\N{figure dash}',  # triple bond, must be followed by space
    r'\\sim': '\N{tilde operator}',
    r'\\simeq': '\N{approximately equal to}',
    r'\\infty': '\N{infinity}',
    r'\\times': '\N{n-ary times operator}',
    r'+-': '\N{plus-minus sign}',
    r'-+': '\N{minus-or-plus sign}',
    r'\\square': '\N{black medium square}',
    r'\\neq': '\N{not equal to}',
    r'\\rangle': '\N{mathematical right angle bracket}',
    r'\\langle': '\N{mathematical left angle bracket}',
    r'\\rightarrow': '\N{rightwards arrow}',
    r'\\leftarrow': '\N{leftwards arrow}',
}
