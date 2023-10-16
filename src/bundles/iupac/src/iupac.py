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
iupac: IUPAC fetch support
"""

from chimerax.core.fetch import fetch_file
from chimerax.core.errors import LimitationError
class IupacTranslationError(LimitationError):
    pass

def fetch_iupac(session, iupac_name, *, res_name=None, **kw):

    printables = []
    for char in iupac_name:
        #if char.isprintable() and not char.isspace():
        if char.isprintable():
            printables.append(char)
    diff = len(iupac_name) - len(printables)
    if diff > 0:
        session.logger.warning("Removed %d blank/non-printable characters from IUPAC name"
            % diff)
        iupac_name = "".join(printables)
    # prevent IUPAC characters from getting mangled by the http protocol
    from urllib.parse import quote
    web_iupac = quote(iupac_name)
    for fetcher, moniker, ack_name, info_url, user_url in fetcher_info:
        try:
            smiles = fetcher(session, iupac_name, web_iupac)
        except IupacTranslationError:
            pass
        else:
            from chimerax.smiles import fetch_smiles
            structures, status = fetch_smiles(session, smiles, res_name=res_name)
            for s in structures:
                s.name = "iupac:" + iupac_name
            return structures, "Translated IUPAC name to SMILES string via %s web service (IUPAC: %s)\n" % (
                moniker, iupac_name) + status
    raise IupacTranslationError(
        "Web services failed to translate IUPAC name to SMILES string.\n"
        "For more information about the failure, go to %s and input the IUPAC name by hand." % user_url
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
