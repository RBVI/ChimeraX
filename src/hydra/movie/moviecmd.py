from ..commands.parse import CommandError

def movie_command(cmdname, args, session):

    if getattr(session, 'ignore_movie_commands', False):
        a0 = (args.split() + [''])[0]
        if not 'ignore'.startswith(a0):
            session.show_status('Ignoring command: %s %s' % (cmdname, args))
            return

    from ..commands.parse import perform_operation, string_arg, int_arg, ints_arg
    from ..commands.parse import bool_arg, float_arg, enum_arg
    from .movie import RESET_CLEAR, RESET_KEEP, RESET_NONE
    reset_modes = (RESET_CLEAR, RESET_KEEP, RESET_NONE)
    ops = {
        'record': (record_op,
                   (),
                   (),
                   (('directory', string_arg),
                    ('pattern', string_arg),
                    ('format', string_arg),
                    ('fformat', string_arg),    # Obsolete
                    ('size', ints_arg),
                    ('supersample', int_arg),
                    ('limit', int_arg),)),

        'encode': (encode_multiple_op,
                   (),
                   (('output', string_arg, 'multiple'),),
                   (('format', string_arg),
                    ('quality', string_arg),
                    ('qscale', int_arg),
                    ('bitrate', float_arg),
                    ('framerate', float_arg),
                    ('preset', string_arg),
                    ('resetMode', enum_arg, {'values': reset_modes}),
                    ('roundTrip', bool_arg),
                    ('wait', bool_arg),
                    ('mformat', string_arg),    # Obsolete
                    ('buffersize', float_arg),  # Obsolete
                    )),
        'crossfade': (crossfade_op, (), (('frames', int_arg),), ()),
        'duplicate': (duplicate_op, (), (('frames', int_arg),), ()),
        'stop': (stop_op, (), (), ()),
        'abort': (abort_op, (), (), ()),
        'reset': (reset_op,
                  (),
                  (('resetMode', enum_arg, {'values': reset_modes}),),
                  ()),
        'status': (status_op, (), (), ()),
        'formats': (formats_op, (), (), ()),
        'ignore': (ignore_op, (), (('ignore', bool_arg),), ()),
        }

    perform_operation(cmdname, args, ops, session)

def record_op(directory = None, pattern = None, format = None, fformat = None,
              size = None, supersample = 1, limit = 15000, session = None):

    if format is None and fformat:
        format = fformat        # Historical option name.
    
    from . import formats
    if format is None:
        format = formats.default_image_format
    else:
        fmts = formats.image_formats
        format = format.upper()
        if not format in fmts:
            raise CommandError('Unsupported image file format %s, use %s'
                             % (format, ', '.join(fmts)))

    from os.path import isdir, expanduser
    if directory and not isdir(expanduser(directory)):
        raise CommandError('Directory %s does not exist' % (directory,))
    if pattern and pattern.count('*') != 1:
        raise CommandError('Pattern must contain exactly one "*"')

    if not size is None and len(size) != 2:
        raise CommandError('Size must be two comma-separated integers')

    if not supersample is None and supersample < 0:
        raise CommandError('Supersample must be a positive integer')

    movie = getattr(session, 'movie', None)
    if movie is None:
        from .movie import Movie
        session.movie = movie = Movie(format, directory, pattern, size, supersample, limit, session)
    elif movie.is_recording():
        raise CommandError("Already recording a movie")
    else:
        movie.supersample = supersample
        movie.limit = limit

    movie.start_recording()

def encode_multiple_op(session, **kw):

    if 'output' in kw:
        kw1 = kw.copy()
        outputs = kw1.pop('output')
        from .movie import RESET_CLEAR, RESET_KEEP, RESET_NONE
        r = kw1.pop('resetMode') if 'resetMode' in kw1 else RESET_CLEAR
        w = kw1.pop('wait') if 'wait' in kw1 else False
        for o in outputs[:-1]:
            encode_op(output = o, resetMode = RESET_NONE, wait = True, **kw1)
        encode_op(output = outputs[-1], resetMode = r, wait = w, session = session, **kw1)
    else:
        encode_op(session = session, **kw)
    
from .movie import RESET_CLEAR
def encode_op(output=None, format=None,
              quality=None, qscale=None, bitrate=None,
              framerate=25, roundTrip=False, preset=None,
              resetMode=RESET_CLEAR, wait=False,
              mformat=None, buffersize=None, session = None):

    from . import formats

    output_size = None
    bit_rate = None
    qual = None
    if output:
        from os import path
        output = path.expanduser(output)
    if preset:
        preset = preset.upper()
        settings = formats.standard_formats.get(preset)
        if settings is None:
            presets = ', '.join(formats.standard_formats.keys())
            error = 'Unrecognized preset "%s" (%s)' % (preset, presets)
            raise CommandError(error)
        f = settings['format']
        output_size = '%dx%d' % settings['resolution']
        bit_rate = settings['bit_rate']
    else:
        if format is None:
            if mformat:
                format = mformat
            elif output:
                format = format_from_file_suffix(output)
        if format is None:
            fmt_name = formats.default_video_format
        elif format.lower() in formats.formats:
            fmt_name = format.lower()
        else:
            raise CommandError('Unrecognized movie format %s' % format)
        f = formats.formats[fmt_name]
        if bitrate is None and qscale is None and quality is None:
            quality = formats.default_quality
        if quality:
            qopt = f['ffmpeg_quality']
            qual = (qopt['option_name'], qopt[quality])
        elif qscale:
            qual = ('-qscale:v', qscale)
        elif bitrate:
            bit_rate = bitrate

    if output is None:
        import os.path
        ext = f['suffix']
        from .movie import DEFAULT_OUTFILE
        output = '%s.%s' % (os.path.splitext(DEFAULT_OUTFILE)[0], ext)

    movie = getattr(session, 'movie', None)
    if movie is None:
        raise CommandError('No frames have been recorded')
    if movie.is_recording():
        movie.stop_recording()

    movie.start_encoding(output, f['ffmpeg_name'], output_size, f['ffmpeg_codec'], "yuv420p", f['size_restriction'],
                         framerate, bit_rate, qual, roundTrip, resetMode)

def crossfade_op(session, frames=25):

    session.movie.postprocess('crossfade', frames)

def duplicate_op(session, frames=25):

    session.movie.postprocess('duplicate', frames)

def stop_op(session):

    session.movie.stop_recording()

def abort_op(session):

    session.movie.stop_encoding()

def reset_op(session, resetMode = RESET_CLEAR):

    clr = (resetMode == RESET_CLEAR)
    session.movie.resetRecorder(clearFrames=clr)

def ignore_op(session, ignore = True):

    session.ignore_movie_commands = ignore

def status_op(session):

    session.movie.dumpStatusInfo()

def formats_op(session):

    from . import formats
    flist = '\n'.join('\t%s\t=\t %s (.%s)' % (n, f['label'], f['suffix'])
                      for name, f in formats.formats.items())
    fnames = ' '.join(f for f in formats.formats.keys())
    session.show_info('Movie encoding formats:\n%s\n' % flist)
    session.show_status('Movie formats: %s' % fnames)

def format_from_file_suffix(path):

    from . import formats
    for name,f in formats.formats.items():
        suffix = '.' + f['suffix']
        if path.endswith(suffix):
            return name
    return None

def command_keywords():

    rec_args = ('directory', 'pattern', 'format', 'fformat', 'size',
                'supersample', 'limit')
    enc_args = ('output', 'format', 'quality', 'bitrate', 'framerate',
                'preset', 'resetMode', 'roundTrip', 'wait',
                'mformat', 'buffersize')
    cr_args = ('frames',)
    return rec_args + enc_args + cr_args

def wait_command(cmd_name, args, session):

    from ..commands.parse import int_arg, parse_arguments
    req_args = ()
    opt_args = (('frames', int_arg),)
    kw_args = ()

    kw = parse_arguments(cmd_name, args, session, req_args, opt_args, kw_args)
    kw['session'] = session
    wait(**kw)

def wait(frames = None, session = None):

    v = session.view
    if frames is None:
        from ..commands.motion import motion_in_progress
        while motion_in_progress(session):
            v.redraw_needed = True  # Trigger frame rendered callbacks to cause image capture.
            v.redraw_graphics()
    else:
        for f in range(frames):
            v.redraw_needed = True  # Trigger frame rendered callbacks to cause image capture.
            v.redraw_graphics()
