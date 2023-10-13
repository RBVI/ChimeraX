# vim: set expandtab ts=4 sw=4:

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

# -----------------------------------------------------------------------------
# Fetch density maps from the Electron Microscopy Data Bank
#
#       ftp://ftp.wwpdb.org/pub/emdb/structures/EMD-5582/map/emd_5582.map.gz
#	https://files.rcsb.org/pub/emdb/structures/EMD-1013/map/emd_1013.map.gz
#       ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-5680/map/emd_5680.map.gz
#
def fetch_emdb(session, emdb_id, mirror = None, transfer_method = None, fits = False,
               ignore_cache=False, **kw):
    from chimerax.core.errors import UserError
    if len(emdb_id) < 4:
        raise UserError("EMDB identifiers are at least 4 characters long")

    if mirror is None:
        import socket
        hname = socket.gethostname()
        if hname.endswith('.edu') or hname.endswith('.gov'):
            mirror = 'united states'
        elif hname.endswith('.cn'):
            mirror = 'china'
        elif hname.endswith('.jp'):
            mirror = 'japan'
        else:
            mirror = 'europe'

    # Choice of ftp vs https based on speed tests.  Ticket #5448
    xml_url_pattern = None
    if mirror == 'united states':
        # The RCSB ftp does not report file size so progress messages don't indicate how long it will take.
        if transfer_method == 'ftp':
            url_base = 'ftp://ftp.wwpdb.org/pub/emdb/structures/EMD-%s'
        else:
            url_base = 'https://files.wwpdb.org/pub/emdb/structures/EMD-%s'
    elif mirror == 'china':
        if transfer_method == 'https':
            url_base = 'https://ftp.emdb-china.org/structures/EMD-%s'
        else:
            url_base = 'ftp://ftp.emdb-china.org/structures/EMD-%s'
    elif mirror == 'japan':
        if transfer_method == 'ftp':
            url_base = 'ftp://ftp.pdbj.org/pub/emdb/structures/EMD-%s'
        else:
            url_base = 'https://ftp.pdbj.org/pub/emdb/structures/EMD-%s'
    else:
        if transfer_method == 'https':
            url_base = 'https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-%s'
        else:
            url_base = 'ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-%s'

    url_pattern = url_base + '/map/%s.gz'
    map_name = 'emd_%s.map' % emdb_id
    map_url = url_pattern % (emdb_id, map_name)

    from chimerax.core.fetch import fetch_file
    filename = fetch_file(session, map_url, 'map %s' % emdb_id, map_name, 'EMDB',
                          uncompress = True, ignore_cache=ignore_cache)

    # Use EMDB meta data
    try:
        xml_filename = _emdb_meta_data_xml_file(session, emdb_id, url_base, ignore_cache=ignore_cache)
        meta_data = _emdb_meta_data(xml_filename)
    except Exception as e:
        session.logger.warning(f'Could not read metadata for EMDB {emdb_id}: {str(e)}')
        meta_data = None
        
    # Set initial contour level
    if meta_data and meta_data.contour_levels:
        kw['initial_surface_level'] = meta_data.contour_levels[0]

    model_name = 'emdb %s' % emdb_id
    models, status = session.open_command.open_data(filename, format = 'ccp4',
                                                    name = model_name, **kw)

    # Report fit pdb ids in log message
    pdb_ids = meta_data.pdb_ids if meta_data else None
    if pdb_ids:
        for v in models:
            v.fit_pdb_ids = pdb_ids

    # Open fit atomic models
    if fits:
        if pdb_ids:
            pdb_models, pdb_status = _open_pdbs(session, pdb_ids, mirror=mirror, ignore_cache=ignore_cache)
            models.extend(pdb_models)
        else:
            status += '\nNo fit atomic models.'
        
    return models, status

def fetch_emdb_europe(session, emdb_id, transfer_method = None, fits = False, ignore_cache=False, **kw):
    return fetch_emdb(session, emdb_id, mirror = 'europe', transfer_method = transfer_method,
                      fits = fits, ignore_cache = ignore_cache, **kw)

def fetch_emdb_japan(session, emdb_id, transfer_method = None, fits = False, ignore_cache=False, **kw):
    return fetch_emdb(session, emdb_id, mirror = 'japan', transfer_method = transfer_method,
                      fits = fits, ignore_cache = ignore_cache, **kw)

def fetch_emdb_china(session, emdb_id, transfer_method = None, fits = False, ignore_cache=False, **kw):
    return fetch_emdb(session, emdb_id, mirror = 'china', transfer_method = transfer_method,
                      fits = fits, ignore_cache = ignore_cache, **kw)

def fetch_emdb_us(session, emdb_id, transfer_method = None, fits = False, ignore_cache=False, **kw):
    return fetch_emdb(session, emdb_id, mirror = 'united states', transfer_method = transfer_method,
                      fits = fits, ignore_cache = ignore_cache, **kw)

def fetch_emdb_fits(session, emdb_id, transfer_method = None, fits = True, ignore_cache=False, **kw):
    return fetch_emdb(session, emdb_id, mirror = 'united states', transfer_method = transfer_method,
                      fits = fits, ignore_cache = ignore_cache, **kw)

def _emdb_meta_data_xml_file(session, emdb_id, url_base, ignore_cache = False):
    xml_url_pattern = url_base + '/header/emd-%s.xml'
    xml_name = 'emdb-%s.xml' % emdb_id
    xml_url = xml_url_pattern % (emdb_id, emdb_id)

    from chimerax.core.fetch import fetch_file
    xml_filename = fetch_file(session, xml_url, 'map header %s' % emdb_id, xml_name, 'EMDB',
                              ignore_cache=ignore_cache)
    return xml_filename

class _emdb_meta_data:
    def __init__(self, xml_filename):
        from xml.dom.minidom import parse
        d = parse(xml_filename)
        # Look for fit PDB files.
        self.pdb_ids = _get_xml_tree_values(d, ('emd', 'crossreferences', 'pdb_list', 'pdb_reference', 'pdb_id'))
        self.contour_levels = _get_xml_tree_values(d, ('emd', 'map', 'contour_list', 'contour', 'level'),
                                                   as_type = float)

def _get_xml_tree_values(node, path, as_type = str):
    if len(path) == 0:
        return [as_type(c.data) for c in node.childNodes if c.nodeType == c.TEXT_NODE]
    values = []
    for c in node.childNodes:
        if c.nodeType == c.ELEMENT_NODE and c.tagName == path[0]:
              values.extend(_get_xml_tree_values(c, path[1:], as_type=as_type))
    return values

def _open_pdbs(session, pdb_ids, mirror = 'rcsb', ignore_cache = False):
    fetch_source = {'united states':'rcsb', 'japan':'pdbj', 'europe':'pdbe'}.get(mirror, 'rcsb')
    models = []
    status = []
    for pdb_id in pdb_ids:
        from chimerax.mmcif import fetch_mmcif
        pdb_models, pdb_status = fetch_mmcif(session, pdb_id, fetch_source=fetch_source,
                                             ignore_cache=ignore_cache)
        models.extend(pdb_models)
        status.append(pdb_status)

    if len(pdb_ids) == 0:
        session.logger.info('No fit atomic models.')

    return models, '\n'.join(status)
