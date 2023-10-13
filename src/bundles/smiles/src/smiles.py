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
smiles: SMILES fetch support
"""

from chimerax.core.fetch import fetch_file
from chimerax.core.errors import LimitationError
class SmilesTranslationError(LimitationError):
    pass

def fetch_smiles(session, smiles_string, *, res_name=None, **kw):

    printables = []
    for char in smiles_string:
        if char.isprintable() and not char.isspace():
            printables.append(char)
    diff = len(smiles_string) - len(printables)
    if diff > 0:
        session.logger.warning("Removed %d blank/non-printable characters from SMILES string"
            % diff)
        smiles_string = "".join(printables)
    # many SMILES characters get mangled by the http protocol, so switch to http-friendly equivalent
    from urllib.parse import quote
    web_smiles = quote(smiles_string)
    for fetcher, moniker, ack_name, info_url in fetcher_info:
        try:
            path = fetcher(session, smiles_string, web_smiles)
        except SmilesTranslationError:
            pass
        else:
            from chimerax.sdf import read_sdf
            from chimerax import io
            structures, status = read_sdf(session, io.open_input(path, encoding='utf=8'), path)
            if structures:
                for s in structures:
                    s.name = "smiles:" + smiles_string
                break
        session.logger.info("Failed to translate SMILES to 3D structure via %s web service"
            "(SMILES: %s)" % (moniker, smiles_string))
    else:
        raise SmilesTranslationError(
            "Web services failed to translate SMILES string to 3D structure.")
    translation_info = "Translated SMILES to 3D structure via %s web service (SMILES: %s)" % (
        moniker, smiles_string)
    if res_name is not None:
        for s in structures:
            s.residues.names = res_name
    return structures, translation_info

def _cactus_fetch(session, smiles, web_smiles):
    cactus_site = "cactus.nci.nih.gov"
    from chimerax.io import open_input
    from urllib.error import URLError
    try:
        reply = open_input("http://%s/cgi-bin/translate.tcl?smiles=%s&format=sdf&astyle=kekule"
            "&dim=3D&file=" % (cactus_site, web_smiles), session.data_formats['sdf'].encoding)
    except URLError as e:
        pass
    else:
        for line in reply:
            if "Click here" in line and line.count('"') == 2 and "href=" in line:
                pre, url, post = line.split('"')
                return "http://%s%s" % (cactus_site, url)
    raise SmilesTranslationError("Cactus could not translate %s" % smiles)

def _indiana_fetch(session, smiles, web_smiles):
    from chimerax.core.fetch import fetch_file
    import os
    # use only first 128 characters of SMILES for file name due to Windows file-name size limitations [#9170]
    filename = fetch_file(session, "http://cheminfov.informatics.indiana.edu/rest/thread/d3.py/"
        "SMILES/%s" % smiles, 'SMILES %s' % smiles, web_smiles[:128], None)
    return filename

fetcher_info = [
    (_cactus_fetch, "NCI", "NCI CADD Group", "http://cactus.nci.nih.gov"),
]
