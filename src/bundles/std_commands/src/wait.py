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

def wait(session, frames=None):
    '''Wait before proceeding to the next command. Used in movie recording scripts.

    Parameters
    ----------
    frames : integer
       Wait until this many frames have been rendered before executing the next
       command in a command script.
    '''
    if frames is None:
        from chimerax.core.commands.motion import motion_in_progress
        if not motion_in_progress(session):
            session.logger.warning('wait requires a frame count argument unless motion is in progress')
            return
        while motion_in_progress(session):
            draw_frame(session)
    else:
        for f in range(frames):
            draw_frame(session)

def draw_frame(session, limit_frame_rate = True):
    '''Draw a graphics frame.'''
    if limit_frame_rate:
        from time import time, sleep
        t0 = time()

    v = session.main_view
    v.redraw_needed = True  # Trigger frame rendered callbacks to cause image capture.
    ul = session.update_loop
    ul.draw_new_frame()

    if limit_frame_rate and session.ui.is_gui:
        dt = time() - t0
        frame_time= session.update_loop.redraw_interval / 1000.0	# seconds
        if dt < frame_time:
            sleep(frame_time - dt)


def register_command(logger):
    from chimerax.core.commands import CmdDesc, PositiveIntArg, register
    desc = CmdDesc(
        optional=[('frames', PositiveIntArg)],
        synopsis='suspend command processing for a specified number of frames'
        ' or until finite motions have stopped ')
    register('wait', desc, wait, logger=logger)
