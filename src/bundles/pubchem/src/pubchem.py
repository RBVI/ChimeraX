# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
pubchem: PubChem fetch support
"""

def fetch_pubchem(session, pubchem_id, *, ignore_cache=False, res_name=None, **kw):
    from chimerax.core.errors import UserError
    if not pubchem_id.isdigit():
        raise UserError('PubChem identifiers are numeric, got "%s"' % pubchem_id)

    import os
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/%s/SDF?record_type=3d" % pubchem_id
    pubchem_name = "%s.sdf" % pubchem_id
    from chimerax.core.fetch import fetch_file
    filename = fetch_file(session, url, 'PubChem %s' % pubchem_id, pubchem_name,
        'PubChem', ignore_cache=ignore_cache)

    session.logger.status("Opening PubChem %s" % (pubchem_id,))
    structures, status = session.open_command.open_data(filename, format='sdf',
        name="pubchem:" + pubchem_id, **kw)
    if res_name is not None:
        for s in structures:
            s.residues.names = res_name
    return structures, status
