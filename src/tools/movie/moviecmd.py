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
    register('movie record', record_desc, movie_record)

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
    register('movie encode', encode_desc, movie_encode)

    crossfade_desc = CmdDesc(optional = [('frames', IntArg)])
    register('movie crossfade', crossfade_desc, movie_crossfade)

    duplicate_desc = CmdDesc(optional = [('frames', IntArg)])
    register('movie duplicate', duplicate_desc, movie_duplicate)

    stop_desc = CmdDesc()
    register('movie stop', stop_desc, movie_stop)

    abort_desc = CmdDesc()
    register('movie abort', abort_desc, movie_abort)

    reset_desc = CmdDesc(optional = [('reset_mode', EnumOf(reset_modes))])
    register('movie reset', reset_desc, movie_reset)

    status_desc = CmdDesc()
    register('movie status', status_desc, movie_status)

    formats_desc = CmdDesc()
    register('movie formats', formats_desc, movie_formats)

    ignore_desc = CmdDesc(optional = [('ignore', BoolArg)])
    register('movie ignore', ignore_desc, movie_ignore)

from .movie import RESET_CLEAR
def movie_record(session, directory = None, pattern = None, format = None,
                 size = None, supersample = 1, limit = 15000):
    '''Start recording a movie.

    Parameters
    ----------
    directory : string
      A temporary directory for saving image files before the movie is encoded.
      If a directory is specified, it must already exist -- it will not be created.
      If no directory is specified a temporary system directory is created.
    pattern : string
      File name including a "*" character that is substituted with the frame number
      when saving images.
    format : string
      Image file format (default ppm) for saving frames. Possible values ppm, png, jpeg.
      ppm is fastest but takes the most disk space because it is not compressed.
    size : 2 int
      Width and height in pixels of movie.
    supersample : int
      Amount of supersampling when saving individual image frames.
    limit : int
      Maximum number of frames to save.  This is a safe guard so that the entire computer disk storage
      is not filled with images if a movie recording is never stopped.
    '''
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
    return ignore

def movie_encode(session, output=None, format=None, quality=None, qscale=None, bitrate=None,
                 framerate=25, round_trip=False, reset_mode=RESET_CLEAR, wait=False):
    '''Enode images captured with movie record command creating a movie file.
    This uses the standalone video encoding program ffmpeg which is included with Chimera.
    
    Parameters
    ----------
    output : list of strings
      Filenames of movie output files.  By specifying multiple files, different video encoding
      formats can be made.
    format : string
      Format of video file to write.  If not specified, the file suffix determines the format.
      Use of a *.mp4 file suffix and h264 format is by far the best choice.
      Allowed formats (file suffix) are: h264 (*.mp4), mov (*.mov), avi (.avi), wmv (.wmv).
    quality : string
      Quality of video, higher quality results in larger file size.  Qualities are "highest",
      "higher", "high", "good", "medium", "fair", "low".  Default "good".  This overrides
      the qscale and bitrate options.
    qscale : int
      Quality scale parameter used by some video codecs.  This overrides the bitrate option.
    bitrate : int
      Target bit rate for video encoding in Kbits/second.
    framerate : int
      Frames per second that video should playback at in a video player.
    round_trip : bool
      If true, the images are played forward and than backward so movie ends where
      it began.  This is used for making movies that loop without a jump between the
      last frame and first frame.
    reset_mode : string
      Whether to keep or delete the image files that were captured after the video files is made.
      Values are "clear", "keep" or "none".  Default "clear" means the image files are deleted.
    wait : bool
      Whether to wait until movie encoding is finished before the command returns.
      Default false means the command can return before the video encoding is complete.
    '''
    if ignore_movie_commands(session):
        return

    if not output is None:
        from .movie import RESET_NONE
        for o in output[:-1]:
            encode_op(session, o, format, quality, qscale, bitrate,
                      framerate, round_trip, reset_mode = RESET_NONE, wait = True)
        encode_op(session, output[-1], format, quality, qscale, bitrate,
                  framerate, round_trip, reset_mode, wait)
    else:
        encode_op(session, output, format, quality, qscale, bitrate,
                  framerate, round_trip, reset_mode, wait)
    
def encode_op(session, output=None, format=None, quality=None, qscale=None, bitrate=None,
              framerate=25, round_trip=False, reset_mode=RESET_CLEAR, wait=False):

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

def movie_crossfade(session, frames=25):
    '''Linear interpolate between the current graphics image and the next image
    over a specified number of frames.
    '''
    if ignore_movie_commands(session):
        return
    session.movie.postprocess('crossfade', frames)

def movie_duplicate(session, frames=25):
    '''Repeat the current image frame a specified number of frames
    so that the video does not change during playback.'''
    if ignore_movie_commands(session):
        return
    session.movie.postprocess('duplicate', frames)

def movie_stop(session):
    '''Stop recording video frames. Using the movie encode command also stops recording.'''
    if ignore_movie_commands(session):
        return
    session.movie.stop_recording()

def movie_abort(session):
    '''Stop movie recording and delete any recorded frames.'''
    if ignore_movie_commands(session):
        return
    session.movie.stop_encoding()

def movie_reset(session, reset_mode = RESET_CLEAR):
    '''Clear images saved with movie record.'''
    if ignore_movie_commands(session):
        return
    clr = (reset_mode == RESET_CLEAR)
    if hasattr(session, 'movie'):
        session.movie.resetRecorder(clearFrames=clr)

def movie_ignore(session, ignore = True):
    '''Ignore subsequent movie commands except for the movie ignore command.
    This can be used to run a movie recording script without recording
    the movie.'''
    session.replace_attribute('ignore_movie_commands', ignore)

def ignore_movie_commands(session):
    ignore = getattr(session, 'ignore_movie_commands', False)
    return ignore

def movie_status(session):
    '''Report recording status such as number of frames saved to the log.'''
    if ignore_movie_commands(session):
        return
    session.movie.dumpStatusInfo()

def movie_formats(session):
    '''Report the available video formats to the log.'''
    from . import formats
    flist = '\n'.join('\t%s\t=\t %s (.%s)' % (name, f['label'], f['suffix'])
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
