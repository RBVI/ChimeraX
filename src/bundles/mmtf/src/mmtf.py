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
mmtf: MMTF format support
==========================

Fetch and read MMTF format files
"""

from chimerax.core.errors import UserError

def fetch_mmtf(session, pdb_id, ignore_cache=False, **kw):
    if len(pdb_id) != 4:
        raise UserError("PDB identifers are 4 characters long, got %r" % pdb_id)

    pdb_id = pdb_id.lower()
    mmtf_name = '%s.mmtf' % pdb_id

    url = 'https://mmtf.rcsb.org/v1.0/full/%s.mmtf.gz' % pdb_id.upper()
    from chimerax.core.fetch import fetch_file
    filename = fetch_file(session, url, 'MMTF %s' % pdb_id, mmtf_name, 'PDB',
        ignore_cache=ignore_cache, uncompress=True)

    session.logger.status("Opening MMTF %s" % (pdb_id,))
    return session.open_command.open_data(filename, format='mmtf', name=pdb_id, **kw)

def open_mmtf(session, filename, name, auto_style=True, coordsets=False):
    """Create atomic structures from MMTF file

    :param filename: either the name of a file or a file-like object
    """

    if hasattr(filename, 'name'):
        # it's really a fetched stream
        filename = filename.name

    from . import _mmtf
    pointers = _mmtf.parse_MMTF_file(filename, session.logger, coordsets)

    from chimerax.atomic.structure import AtomicStructure
    models = [AtomicStructure(session, name=name, c_pointer=p, auto_style=auto_style) for p in pointers]
    for m in models:
        m.filename = filename

    info = "Opened MMTF data containing %d atoms%s %d bonds" % (
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
                from chimerax.core.commands.coordset import coordset_slider
                coordset_slider(session, mc)
    return models, info
