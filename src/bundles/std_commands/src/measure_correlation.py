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

# -----------------------------------------------------------------------------
#
def measure_correlation(session, map, in_map = None, envelope = True):
    if envelope and map.minimum_surface_level is None:
        from chimerax.core.errors import UserError
        raise UserError('Cannot compute correlation above surface level of %s which has no surface'
                        % map.name_with_id())

    from chimerax.map_fit import map_overlap_and_correlation
    overlap, correlation, correlation_about_mean = \
        map_overlap_and_correlation(map, in_map, above_threshold = envelope)

    log = session.logger
    heading = 'Correlation of %s' % map.name_with_id()
    if envelope:
        heading += '  above level %.4g' % map.minimum_surface_level
    heading += ' in %s' % in_map.name_with_id()
    values = ('correlation = %.4g, correlation about mean = %.4g'
              % (correlation, correlation_about_mean))
    log.status(values)
    log.info( '%s\n%s' % (heading, values))

    return correlation, correlation_about_mean

# -----------------------------------------------------------------------------
#
def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, BoolArg
    from chimerax.map import MapArg
    desc = CmdDesc(
        required = [('map', MapArg)],
        keyword = [('in_map', MapArg),
                   ('envelope', BoolArg)],
        required_arguments = ['in_map'],
        synopsis = 'report the correlation coefficient between two maps')
    register('measure correlation', desc, measure_correlation, logger=logger)
