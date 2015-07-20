#
# perframe command to execute specified command each frame, often for movie recording.
#
# perframe  command  [ range  start,end[,step] ]... [ frames N ] [ interval K ]
#                    [ zeroPadWidth width] [ showCommands ]
#
def register_perframe_command():

    from chimera.core.cli import CmdDesc, register, IntArg, StringArg, NoArg
    desc = CmdDesc(required = [('command', StringArg)],
                   keyword = [('range', RangeArg),	# TODO: Allow multiple range arguments.
                              ('frames', IntArg),
                              ('interval', IntArg),
                              ('zero_pad_width', IntArg),
                              ('show_commands', NoArg)])
    register('perframe', desc, perframe)

def perframe(session, command, frames = None, interval = 1, zero_pad_width = None,
             range = None, show_commands = False):

    if command == 'stop':
        stop_perframe_callbacks(session)
        return

    data = {
        'command': command,
        'frames': frames,
        'interval': interval,
        'skip': 0,
        'ranges': [] if range is None else [range],
        'zero_pad_width': zero_pad_width,
        'show_commands': show_commands,
        'frame_num': 1,
    }
    def cb(data = data, session = session):
        _perframe_callback(data, session)
    data['callback'] = cb
    v = session.main_view
    v.add_new_frame_callback(cb)
    if not hasattr(session, 'perframe_callbacks'):
        session.perframe_callbacks = set()
    session.perframe_callbacks.add(cb)

def _perframe_callback(data, session):
    d = data
    if d['interval'] > 1:
        if d['skip'] > 0:
            d['skip'] -= 1
            return
        else:
            d['skip'] = d['interval'] - 1

    frames, frame_num = d['frames'], d['frame_num']
    cmd, stop = _expand_command(d['command'], frame_num, frames, d['ranges'], d['zero_pad_width'])
    if d['show_commands']:
        session.logger.info('perframe %d: %s\n' % (frame_num, cmd))
    try:
        cmd = cli.Command(session, cmd, final=True)
        cmd.execute()
    except:
        stop_perframe_callbacks(session, [d['callback']])
        import sys
        session.logger.warning("Error executing per-frame command '%s': %s"
                               % (cmd, sys.exc_info()[1]))
        return
    if stop or (not frames is None and frame_num >= frames):
        stop_perframe_callbacks(session, [d['callback']])
    else:
        d['frame_num'] = frame_num + 1

def _expand_command(command, frame_num, frames, ranges, zero_pad_width):
    args, stop = _perframe_args(frame_num, frames, ranges, zero_pad_width)
    for i, arg in enumerate(args):
        var = '$%d' % (i + 1)
        command = command.replace(var, arg)
    return command, stop

def _perframe_args(frame_num, frames, ranges, zero_pad_width):
    args = []
    stop = False
    if ranges:
        fmt = '%%0%dg' % zero_pad_width if zero_pad_width else '%g'
        for vr in ranges:
            r0, r1 = vr[:2]
            explicit_rstep = not vr[2] is None
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
        fmt = '%%0%dd' % zero_pad_width if zero_pad_width else '%d'
        args.append(fmt % frame_num)
    return args, stop

def stop_perframe_callbacks(session, callbacks = None):

    if not hasattr(session, 'perframe_callbacks') and callbacks:
        from . import cli
        raise cli.UserError("No per-frame command active")
    pfcb = session.perframe_callbacks
    if callbacks is None:
        callbacks = tuple(pfcb)
    for cb in callbacks:
        v = session.main_view
        v.remove_new_frame_callback(cb)
        pfcb.remove(cb)

# -----------------------------------------------------------------------------
#
from . import cli
class RangeArg(cli.Annotation):
    '''2 or 3 integers: start, end, step'''
    name = 'range'

    @staticmethod
    def parse(text, session):
        token, text, rest = cli.next_token(text)
        try:
            il = [int(x) for x in token.split(',')]
        except ValueError:
            il = []
        n = len(il)
        if n < 2 or n > 3:
            raise cli.UserError('Range argument must be 2 or 3 comma-separateed integer values, got %s' % token)
        i0,i1 = il[:2]
        step = il[2] if n >= 3 else None
        return (i0,i1,step), text, rest
