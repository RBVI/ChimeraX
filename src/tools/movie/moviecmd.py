from chimera.core.errors import UserError as CommandError
def register_movie_command():

    from chimera.core.commands import CmdDesc, register, BoolArg, EnumOf, ListOf, IntArg, Int2Arg, StringArg, FloatArg

    from .formats import image_formats, formats, qualities
    ifmts = image_formats
    fmts = tuple(formats.keys())
    record_desc = CmdDesc(
        keyword = [('directory', StringArg),
                   ('pattern', StringArg),
                   ('format', EnumOf(ifmts)),
                   ('size', Int2Arg),
                   ('supersample', IntArg),
                   ('limit', IntArg)])
    register('movie record', record_desc, record_op)

    from .movie import RESET_CLEAR, RESET_KEEP, RESET_NONE
    reset_modes = (RESET_CLEAR, RESET_KEEP, RESET_NONE)
    encode_desc = CmdDesc(
        optional = [('output', ListOf(StringArg))],
        keyword = [('format', EnumOf(fmts)),
                   ('quality', EnumOf(qualities)),
                   ('qscale', IntArg),
                   ('bitrate', FloatArg),
                   ('framerate', FloatArg),
                   ('reset_mode', EnumOf(reset_modes)),
                   ('round_trip', BoolArg),
                   ('wait', BoolArg),
               ])
    register('movie encode', encode_desc, encode_multiple_op)

    crossfade_desc = CmdDesc(optional = [('frames', IntArg)])
    register('movie crossfade', crossfade_desc, crossfade_op)

    duplicate_desc = CmdDesc(optional = [('frames', IntArg)])
    register('movie duplicate', duplicate_desc, duplicate_op)

    stop_desc = CmdDesc()
    register('movie stop', stop_desc, stop_op)

    abort_desc = CmdDesc()
    register('movie abort', abort_desc, abort_op)

    reset_desc = CmdDesc(optional = [('reset_mode', EnumOf(reset_modes))])
    register('movie reset', reset_desc, reset_op)

    status_desc = CmdDesc()
    register('movie status', status_desc, status_op)

    formats_desc = CmdDesc()
    register('movie formats', formats_desc, formats_op)

    ignore_desc = CmdDesc(optional = [('ignore', BoolArg)])
    register('movie ignore', ignore_desc, ignore_op)

def record_op(session, directory = None, pattern = None, format = None,
              size = None, supersample = 1, limit = 15000):

    if ignore_movie_commands(session):
        return
    
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
        movie = Movie(format, directory, pattern, size, supersample, limit, session)
        session.replace_attribute('movie', movie)
    elif movie.is_recording():
        raise CommandError("Already recording a movie")
    else:
        movie.supersample = supersample
        movie.limit = limit

    movie.start_recording()

def ignore_movie_commands(session):
    ignore = getattr(session, 'ignore_movie_commands', False)
    if ignore:
        session.logger.info('Ignoring command: %s %s' % (cmdname, args))
    return ignore

def encode_multiple_op(session, **kw):

    if 'output' in kw:
        kw1 = kw.copy()
        outputs = kw1.pop('output')
        from .movie import RESET_CLEAR, RESET_KEEP, RESET_NONE
        r = kw1.pop('reset_mode') if 'reset_mode' in kw1 else RESET_CLEAR
        w = kw1.pop('wait') if 'wait' in kw1 else False
        for o in outputs[:-1]:
            encode_op(output = o, reset_mode = RESET_NONE, wait = True, **kw1)
        encode_op(session, output = outputs[-1], reset_mode = r, wait = w, **kw1)
    else:
        encode_op(session, **kw)
    
from .movie import RESET_CLEAR
def encode_op(session, output=None, format=None,
              quality=None, qscale=None, bitrate=None,
              framerate=25, round_trip=False,
              reset_mode=RESET_CLEAR, wait=False):

    from . import formats

    output_size = None
    bit_rate = None
    qual = None
    if output:
        from os import path
        output = path.expanduser(output)

    if format is None:
        if output:
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
                         framerate, bit_rate, qual, round_trip, reset_mode)

def crossfade_op(session, frames=25):

    session.movie.postprocess('crossfade', frames)

def duplicate_op(session, frames=25):

    session.movie.postprocess('duplicate', frames)

def stop_op(session):

    session.movie.stop_recording()

def abort_op(session):

    session.movie.stop_encoding()

def reset_op(session, reset_mode = RESET_CLEAR):

    clr = (reset_mode == RESET_CLEAR)
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
    session.logger.info('Movie encoding formats:\n%s\n' % flist)
    session.logger.status('Movie formats: %s' % fnames)

def format_from_file_suffix(path):

    from . import formats
    for name,f in formats.formats.items():
        suffix = '.' + f['suffix']
        if path.endswith(suffix):
            return name
    return None
