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
iupac: IUPAC fetch support
"""

from chimerax.core.fetch import fetch_file
from chimerax.core.errors import LimitationError
class IupacTranslationError(LimitationError):
    pass

def fetch_iupac(session, iupac_string, *, res_name=None, **kw):

    printables = []
    for char in iupac_string:
        #if char.isprintable() and not char.isspace():
        if char.isprintable():
            printables.append(char)
    diff = len(iupac_string) - len(printables)
    if diff > 0:
        session.logger.warning("Removed %d blank/non-printable characters from IUPAC string"
            % diff)
        iupac_string = "".join(printables)
    # prevent IUPAC characters from getting mangled by the http protocol
    from urllib.parse import quote
    web_iupac = quote(iupac_string)
    for fetcher, moniker, ack_name, info_url, user_url in fetcher_info:
        try:
            smiles = fetcher(session, iupac_string, web_iupac)
        except IupacTranslationError:
            pass
        else:
            from chimerax.smiles import fetch_smiles
            structures, status = fetch_smiles(session, smiles, res_name=res_name)
            for s in structures:
                s.name = "iupac:" + iupac_string
            return structures, "Translated IUPAC to SMILES string via %s web service (IUPAC: %s)\n" % (
                moniker, iupac_string) + status
    raise IupacTranslationError(
        "Web services failed to translate IUPAC string to SMILES string.\n"
        "For more information about the failure, go to %s and input the IUPAC string by hand." % user_url
        )

def _cambridge_fetch(session, iupac, web_iupac):
    from chimerax.io import open_input
    try:
        reply = open_input("https://opsin.ch.cam.ac.uk/opsin/%s.smi" % web_iupac, 'utf-8')
        lines = [l for l in reply]
        if len(lines) != 1:
            raise IOError("Wrong number of lines in reply")
    except Exception:
        raise IupacTranslationError("Cambridge could not translate %s" % iupac)
    return lines[0]

fetcher_info = [
    (_cambridge_fetch, "Cambridge", "University of Cambridge Centre for Molecular Informatics",
        "https://www-cmi.ch.cam.ac.uk", "https://opsin.ch.cam.ac.uk/index.html"),
]
