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

# -----------------------------------------------------------------------------
#
def volume_channels(session, volumes):
    from chimerax.core.errors import UserError
    if len(volumes) <= 1:
        raise UserError('volume channels: Must specify 2 or more maps, got %d' % len(volumes))

    v0 = volumes[0]
    for v in volumes[1:]:
        if tuple(v.data.size) != tuple(v0.data.size):
            raise UserError('volume channels: Maps must have same size,' +
                            ' got %d,%d,%d' % tuple(v0.data.size)  +
                            ' (%s)' % v0.name_with_id() +
                            ' and %d,%d,%d' % tuple(v.data.size) +
                            ' (%s)' % v.name_with_id())

    for v in volumes:
        if v._channels is not None:
            raise UserError('volume channels: Map %s already is a channel' % v.name_with_id())

    session.models.remove(volumes)
    
    from os.path import commonprefix
    prefix = commonprefix([v.name for v in volumes])
    name = (prefix + ' channels') if len(prefix) >= 5 else 'channels'
    
    from .volume import MapChannelsModel
    mc = MapChannelsModel(name, volumes, session)
    session.models.add([mc])

    return mc

# -----------------------------------------------------------------------------
# Volume channels commands
#
def register_volume_channels_command(logger):
    from chimerax.core.commands import CmdDesc, register
    from .mapargs import MapsArg

    desc = CmdDesc(
        required = [('volumes', MapsArg)],
        synopsis = 'Group volumes under a multichannel volume model')
    register('volume channels', desc, volume_channels, logger=logger)
