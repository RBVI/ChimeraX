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

def perframe(session, command, frames = None, interval = 1, format = None,
             zero_pad_width = None, ranges = None, show_commands = False):
    '''Execute specified command each frame, typically used during movie recording.

    Parameters
    ----------
    command : string
      The command to be run each frame, optionally containing "$1"
      which will be replaced by the frame number starting at 0.
    frames : int
      Number of frames to execute the specified command.
    interval : int
      Run the command only every Kth frame.
    format : string
      Printf style format (e.g. %d, %.3f) for substituting value in for $1.
    zero_pad_width : int
      Field width in characters used when substituting $1 left padded with zeros.
    ranges : list of tuples of 2 or 3 int
      start,end[,step] integer or float range of values to substitute for $1 instead of frame number.
    show_commands : bool
      Whether to echo commands to log.
    '''

    if command == 'stop':
        stop_perframe_callbacks(session)
        return

    from chimerax.core.commands import cli
    data = {
        'command': cli.Alias(command),
        'frames': frames,
        'interval': interval,
        'skip': 0,
        'ranges': [] if range is None else ranges,
        'format': format,
        'zero_pad_width': zero_pad_width,
        'show_commands': show_commands,
        'frame_num': 1,
    }
    def cb(*_, data = data, session = session):
        _perframe_callback(data, session)
    data['handler'] = session.triggers.add_handler('new frame', cb)
    if not hasattr(session, 'perframe_handlers'):
        session.perframe_handlers = set()
    session.perframe_handlers.add(data['handler'])

def register_command(logger):

    from chimerax.core.commands import CmdDesc, register, IntArg, StringArg, BoolArg, \
        RepeatOf, create_alias
    desc = CmdDesc(required = [('command', StringArg)],
                   keyword = [('ranges', RepeatOf(RangeArg)),      # TODO: Allow multiple range arguments.
                              ('frames', IntArg),
                              ('interval', IntArg),
                              ('format', StringArg),    # TODO: Allow multiple format arguments.
                              ('zero_pad_width', IntArg),
                              ('show_commands', BoolArg)],
                   synopsis = 'Run a command before each graphics frame is drawn')
    register('perframe', desc, perframe, logger=logger)
    desc = CmdDesc(synopsis = 'Stop all perframe commands')
    register('perframe stop', desc, stop_perframe_callbacks, logger=logger)
    create_alias('~perframe', 'perframe stop')

def _perframe_callback(data, session):
    d = data
    if d['interval'] > 1:
        if d['skip'] > 0:
            d['skip'] -= 1
            return
        else:
            d['skip'] = d['interval'] - 1

    frames, frame_num = d['frames'], d['frame_num']
    args, stop = _perframe_args(frame_num, frames, d['ranges'],
                                d['format'], d['zero_pad_width'])
    tag = None
    if d['show_commands']:
        tag = 'perframe %d: ' % frame_num
    alias = d['command']
    try:
        alias(session, *args, echo_tag=tag, log=False)
    except Exception:
        stop_perframe_callbacks(session, [d['handler']])
        if alias.cmd is not None:
            cmd_text = alias.cmd.current_text
        else:
            cmd_text = alias.original_text
        import sys
        session.logger.warning("Error executing per-frame command '%s': %s"
                               % (cmd_text, sys.exc_info()[1]))
        return
    if stop or (frames is not None and frame_num >= frames):
        stop_perframe_callbacks(session, [d['handler']])
    else:
        d['frame_num'] = frame_num + 1

def _perframe_args(frame_num, frames, ranges, format, zero_pad_width):
    args = []
    stop = False
    if format is not None:
        fmt = format
    elif zero_pad_width is not None:
        fmt = ('%%0%dg' if ranges else '%%0%dd') % zero_pad_width
    else:
        fmt = '%g' if ranges else '%d'
    if ranges:
        for vr in ranges:
            r0, r1 = vr[:2]
            explicit_rstep = len(vr) > 2
            if frames is None or explicit_rstep:
                rstep = (vr[2] if explicit_rstep
                         else (1 if r1 >= r0 else -1))
                v = r0 + rstep * (frame_num - 1)
                v = min(r1, v) if r1 >= r0 else max(r1, v)
                if frames is None and v == r1:
                    stop = True
            else:
                f = (float(frame_num - 1) / (frames - 1) if frames > 1
                     else 1.0)
                v = r0 * (1 - f) + r1 * f
            args.append(fmt % v)
    else:
        args.append(fmt % frame_num)
    return args, stop

def stop_perframe_callbacks(session, handlers = None):

    if not hasattr(session, 'perframe_handlers'):
        from chimerax.core import errors
        raise errors.UserError("No per-frame command active")
    pfh = session.perframe_handlers
    if handlers is None:
        handlers = tuple(pfh)
    for h in handlers:
        session.triggers.remove_handler(h)
        pfh.remove(h)

# -----------------------------------------------------------------------------
#
from chimerax.core.commands import ListOf, FloatArg
RangeArg = ListOf(FloatArg, 2, 3, name='a range')
