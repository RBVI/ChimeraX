# vi: set expandtab shiftwidth=4 softtabstop=4:

def perframe(session, command, frames = None, interval = 1, format = None,
             zero_pad_width = None, range = None, show_commands = False):
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
    range : 2 or 3 int
      start,end[,step] integer or float range of values to substitute for $1 instead of frame number.
    show_commands : bool
      Whether to echo commands to log.
    '''

    if command == 'stop':
        stop_perframe_callbacks(session)
        return

    from . import cli
    data = {
        'command': cli.Alias(command),
        'frames': frames,
        'interval': interval,
        'skip': 0,
        'ranges': [] if range is None else [range],
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

def register_command(session):

    from .cli import CmdDesc, register, IntArg, StringArg, NoArg
    desc = CmdDesc(required = [('command', StringArg)],
                   keyword = [('range', RangeArg),      # TODO: Allow multiple range arguments.
                              ('frames', IntArg),
                              ('interval', IntArg),
                              ('format', StringArg),    # TODO: Allow multiple format arguments.
                              ('zero_pad_width', IntArg),
                              ('show_commands', NoArg)])
    register('perframe', desc, perframe)
    register('~perframe', CmdDesc(), stop_perframe_callbacks)

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
        alias(session, *args, echo_tag=tag)
    except:
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
        from .. import errors
        raise errors.UserError("No per-frame command active")
    pfh = session.perframe_handlers
    if handlers is None:
        handlers = tuple(pfh)
    for h in handlers:
        session.triggers.delete_handler(h)
        pfh.remove(h)

# -----------------------------------------------------------------------------
#
from . import cli
RangeArg = cli.ListOf(cli.FloatArg, 2, 3, name='a range')
