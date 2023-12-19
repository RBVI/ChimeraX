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

from chimerax.core.errors import UserError as CommandError
def register_movie_command(logger):

    from chimerax.core.commands import CmdDesc, register, BoolArg, EnumOf, ListOf, IntArg, Int2Arg, StringArg, FloatArg, SaveFolderNameArg, SaveFileNameArg

    from .formats import image_formats, formats, qualities
    ifmts = image_formats
    fmts = tuple(formats.keys())
    record_desc = CmdDesc(
        keyword = [('directory', SaveFolderNameArg),
                   ('pattern', StringArg),
                   ('format', EnumOf(ifmts)),
                   ('size', Int2Arg),
                   ('supersample', IntArg),
                   ('transparent_background', BoolArg),
                   ('limit', IntArg)],
        synopsis = 'Start saving frames of a movie to image files')
    register('movie record', record_desc, movie_record, logger=logger)

    from .movie import RESET_CLEAR, RESET_KEEP, RESET_NONE
    reset_modes = (RESET_CLEAR, RESET_KEEP, RESET_NONE)
    encode_desc = CmdDesc(
        optional = [('output', ListOf(SaveFileNameArg))],
        keyword = [('format', EnumOf(fmts)),
                   ('quality', EnumOf(qualities)),
                   ('qscale', IntArg),
                   ('bitrate', FloatArg),
                   ('framerate', FloatArg),
                   ('reset_mode', EnumOf(reset_modes)),
                   ('round_trip', BoolArg),
                   ('wait', BoolArg),
                   ('verbose', BoolArg),
               ],
        synopsis = 'Convert image files into a video file')
    register('movie encode', encode_desc, movie_encode, logger=logger)

    crossfade_desc = CmdDesc(optional = [('frames', IntArg)],
                             synopsis = 'Add images to crossfade between current and next frame')
    register('movie crossfade', crossfade_desc, movie_crossfade, logger=logger)

    duplicate_desc = CmdDesc(optional = [('frames', IntArg)],
                             synopsis = 'Duplicate the last frame to create a pause in a movie')
    register('movie duplicate', duplicate_desc, movie_duplicate, logger=logger)

    stop_desc = CmdDesc(synopsis = 'Pause movie recording')
    register('movie stop', stop_desc, movie_stop, logger=logger)

    abort_desc = CmdDesc(synopsis = 'Stop movie recording and delete saved image files')
    register('movie abort', abort_desc, movie_abort, logger=logger)

    reset_desc = CmdDesc(optional = [('reset_mode', EnumOf(reset_modes))],
                         synopsis = 'Specify whether to save image files after movie encoding')
    register('movie reset', reset_desc, movie_reset, logger=logger)

    status_desc = CmdDesc(synopsis = 'Report recording status such as number of frames saved to the log')
    register('movie status', status_desc, movie_status, logger=logger)

    formats_desc = CmdDesc(synopsis = 'Report the available video formats to the log')
    register('movie formats', formats_desc, movie_formats, logger=logger)

    ignore_desc = CmdDesc(optional = [('ignore', BoolArg)],
                          synopsis = 'Ignore subsequent movie commands')
    register('movie ignore', ignore_desc, movie_ignore, logger=logger)

from .movie import RESET_CLEAR
def movie_record(session, directory = None, pattern = None, format = None,
                 size = None, supersample = 1, transparent_background = False,
                 limit = 90000):
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
    transparent_background : bool
      Whether to save images with transparent background.  Default false.
      Only the PNG image format supports this.  None of the movie encoding formats
      support transparent background so this option is only useful to get the
      individual PNG image frames.
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

    if transparent_background:
        session.logger.info('Movie recording will save image frames with transparent background '
                            'but none of the encoded movie formats handle transparency, so only '
                            'this option is only useful to get the individual frame images.')
        if format != 'PNG':
            raise CommandError('Transparent background is only supported with PNG format '
                               'images.  Use movie record option "format png".')

    movie = getattr(session, 'movie', None)
    if movie is None:
        from .movie import Movie
        movie = Movie(format, directory, pattern, size, supersample, transparent_background,
                      limit, False, session)
        session.movie = movie
    elif movie.is_recording():
        raise CommandError("Already recording a movie")
    else:
        movie.supersample = supersample
        movie.transparent_background = transparent_background
        movie.limit = limit

    movie.start_recording()

def ignore_movie_commands(session):
    ignore = getattr(session, 'ignore_movie_commands', False)
    return ignore

def movie_encode(session, output=None, format=None, quality=None, qscale=None, bitrate=None,
                 framerate=25, round_trip=False, reset_mode=RESET_CLEAR, wait=False, verbose=False):
    '''Enode images captured with movie record command creating a movie file.
    This uses the standalone video encoding program ffmpeg which is included with ChimeraX.
    
    Parameters
    ----------
    output : list of strings
      Filenames of movie output files.  By specifying multiple files, different video encoding
      formats can be made.
    format : string
      Format of video file to write.  If not specified, the file suffix determines the format.
      Use of a .mp4 file suffix and h264 format is by far the best choice.
      Allowed formats (file suffix) are: h264 (.mp4), mov (.mov), avi (.avi), wmv (.wmv).
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
                  framerate, round_trip, reset_mode, wait, verbose)
    else:
        encode_op(session, output, format, quality, qscale, bitrate,
                  framerate, round_trip, reset_mode, wait, verbose)
    
def encode_op(session, output=None, format=None, quality=None, qscale=None, bitrate=None,
              framerate=25, round_trip=False, reset_mode=RESET_CLEAR, wait=False, verbose=False):

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
                suffixes = set(fmt['suffix'] for fmt in formats.formats.values())
                sufs = ', '.join('*.%s' % s for s in suffixes)
                from os.path import basename
                raise CommandError('Unrecognized movie file suffix %s, use %s'
                                   % (basename(output), sufs))
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
        if qopt is not None:
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
    movie.verbose = verbose

    movie.start_encoding(output, f['ffmpeg_name'], output_size, f['ffmpeg_codec'], "yuv420p", f['size_restriction'],
                         framerate, bit_rate, qual, round_trip, reset_mode)

def movie_crossfade(session, frames=25):
    '''Linear interpolate between the current graphics image and the next image
    over a specified number of frames.
    '''
    if ignore_movie_commands(session) or no_movie(session):
        return
    session.movie.postprocess('crossfade', frames)

def movie_duplicate(session, frames=25):
    '''Repeat the current image frame a specified number of frames
    so that the video does not change during playback.'''
    if ignore_movie_commands(session) or no_movie(session):
        return
    session.movie.postprocess('duplicate', frames)

def movie_stop(session):
    '''Stop recording video frames. Using the movie encode command also stops recording.'''
    if ignore_movie_commands(session) or no_movie(session):
        return
    session.movie.stop_recording()

def movie_abort(session):
    '''Stop movie recording and delete any recorded frames.'''
    if ignore_movie_commands(session) or no_movie(session):
        return
    session.movie.stop_encoding()

def movie_reset(session, reset_mode = RESET_CLEAR):
    '''Clear images saved with movie record.'''
    if ignore_movie_commands(session) or no_movie(session):
        return
    clr = (reset_mode == RESET_CLEAR)
    session.movie.resetRecorder(clearFrames=clr)

def movie_ignore(session, ignore = True):
    '''Ignore subsequent movie commands except for the movie ignore command.
    This can be used to run a movie recording script without recording
    the movie.'''
    session.ignore_movie_commands = ignore

def ignore_movie_commands(session):
    ignore = getattr(session, 'ignore_movie_commands', False)
    return ignore

def no_movie(session):
    if not hasattr(session, 'movie') or session.movie is None:
        session.logger.warning('No movie being recorded.')
        return True
    return False

def movie_status(session):
    '''Report recording status such as number of frames saved to the log.'''
    if ignore_movie_commands(session) or no_movie(session):
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
