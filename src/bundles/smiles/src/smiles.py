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
smiles: SMILES fetch support
"""

from chimerax.core.fetch import fetch_file
from chimerax.core.errors import LimitationError
class SmilesTranslationError(LimitationError):
    pass

def fetch_smiles(session, smiles_string, **kw):

    printables = []
    for char in smiles_string:
        if char.isprintable() and not char.isspace():
            printables.append(char)
    diff = len(smiles_string) - len(printables)
    if diff > 0:
        session.logger.warning("Removed %d blank/non-printable characters from SMILES string" % diff)
        smiles_string = "".join(printables)
    # triple-bond characters (#) get mangled by the http protocol, so switch to http-friendly equivalent
    web_smiles = smiles_string.replace('#', "%23")
    for fetcher, moniker, ack_name, info_url in fetcher_info:
        try:
            path = fetcher(session, web_smiles)
        except SmilesTranslationError:
            pass
        else:
            from chimerax.atomic.sdf import read_sdf
            from chimerax.core.io import open_filename
            structures, status = read_sdf(session, open_filename(path, url_encoding='utf=8'), path)
            if structures:
                for s in structures:
                    s.name = "smiles:" + smiles_string
                break
        session.logger.info("Failed to translate SMILES to 3D structure via %s wen service (SMILES: %s)"
            % (moniker, smiles_string))
    else:
        raise SmilesTranslationError("Web services failed to translate SMILES string to 3D structure.")
    translation_info = "Translated SMILES to 3D structure via %s web service (SMILES: %s)" % (
        moniker, smiles_string)
    return structures, translation_info

def _cactus_fetch(session, smiles):
    cactus_site = "cactus.nci.nih.gov"
    from chimerax.core.io import open_filename
    from urllib.error import URLError
    try:
        reply = open_filename("http://%s/cgi-bin/translate.tcl?smiles=%s&format=sdf&astyle=kekule&dim=3D"
            "&file=" % (cactus_site, smiles))
    except URLError as e:
        pass
    else:
        for line in reply:
            if "Click here" in line and line.count('"') == 2 and "href=" in line:
                pre, url, post = line.split('"')
                return "http://%s%s" % (cactus_site, url)
    raise SmilesTranslationError("Cactus could not translate %s" % smiles)

def _indiana_fetch(session, smiles):
    from chimerax.core.fetch import fetch_file
    filename = fetch_file(session, "http://cheminfov.informatics.indiana.edu/rest/thread/d3.py/SMILES/%s"
        % smiles, 'SMILES %s' % smiles, smiles, None)
    return filename

fetcher_info = [
    (_cactus_fetch, "NCI", "NCI CADD Group", "http://cactus.nci.nih.gov"),
    (_indiana_fetch, "Indiana University", "CICC@iu",
        "http://www.soic.indiana.edu/faculty-research/chemical-informatics-center.html"),
]
