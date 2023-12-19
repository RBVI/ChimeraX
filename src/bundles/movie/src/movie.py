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

EXIT_ERROR = "ERROR"
EXIT_CANCEL = "CANCEL"
EXIT_SUCCESS = "SUCCESS"

DEFAULT_PATTERN = "chimovie_%s-*"
import os.path
DEFAULT_OUTFILE = os.path.expanduser("~/Desktop/movie.mp4")

RESET_CLEAR = 'clear'
RESET_KEEP = 'keep'
RESET_NONE = 'none'

from chimerax.core.errors import UserError as MovieError

class Movie:

    def __init__(self, img_fmt=None, img_dir=None, input_pattern=None,
                 size=None, supersample=0, transparent_background=False,
                 limit = None, verbose = False, session = None):

        self.session = session
        self.verbose = verbose

        self.img_fmt = "PNG" if img_fmt is None else img_fmt.upper()

        if not img_dir:
            import tempfile
            self.img_dir = tempfile.gettempdir()
        else:
            from os.path import expanduser
            self.img_dir = expanduser(img_dir)
            
        if not input_pattern:
            self.input_pattern = DEFAULT_PATTERN % getRandomChars()
        else:
            if not input_pattern.count("*")==1:
                raise MovieError("Image pattern must have one and only one '*'")
            self.input_pattern = input_pattern

        self.size = size
        if size and supersample == 0:
            supersample = 1
        self.supersample = supersample
        self.transparent_background = transparent_background
        self.limit = limit

        self.newFrameHandle = None

        self.frame_number = -1		# Last captured frame

        self.postprocess_action = None
        self.postprocess_frames = 0

        self.recording = False
        self.task = None
        self.encoder = None
        self.encoding_thread = None
        self.resetMode = None
        self._image_capture_handler = None

#        from chimera import triggers, COMMAND_ERROR, SCRIPT_ABORT
#        triggers.addHandler(COMMAND_ERROR, self.haltRecording, None)
#        triggers.addHandler(SCRIPT_ABORT, self.haltRecording, None)

    def start_recording(self):
        t = self.session.triggers
        self._image_capture_handler = t.add_handler('frame drawn', self.capture_image)
        self.recording = True
#        from chimera.tasks import Task
#        self.task = Task("record movie", self.cancelCB)
        
    def is_recording(self):
        return self.recording

    def cancelCB(self):
        # If user cancels inside of capture_image, this callback
        # is never invoked because of the exception thrown in saveImage
        # happens before this function is called.  So we 
        self.stop_recording()
        self._notifyStatus("movie recording aborted by user")

    def stop_recording(self):
        if not self.is_recording():
            raise MovieError("Not currently recording")
        if self.postprocess_frames:
            self.capture_image()        # Finish crossfade if one is in progress.
        t = self.session.triggers
        t.remove_handler(self._image_capture_handler)
        self.recording = False
#        self.task.finished()
        self.task = None
        v = self.session.main_view
        if hasattr(v, 'movie_image_rgba'):
            delattr(v, 'movie_image_rgba')
        
    def reset(self):
        self.frame_number = -1
        self.img_dir = None
        self.img_fmt = None
        self.input_pattern = None

    def clearFrames(self):
        import os.path
        src_img_pattern = os.path.join(self.img_dir,
                                       self.input_pattern \
                                       + ".%s" % self.img_fmt.lower()
                                       )
        
        import os, glob
        src_img_paths = glob.glob(src_img_pattern)

        for s in src_img_paths:
            try:
                os.remove(s)
            except Exception:
                self.session.logger.info("Error removing file %s" % s)


    def getFrameCount(self):
        return self.frame_number + 1

    def getInputPattern(self):
        return self.input_pattern

    def getImgFormat(self):
        return self.img_fmt

    def getImgDir(self):
        return self.img_dir

    def getStatusInfo(self):
        status_str  =  "-----Movie status------------------------------\n "
        status_str  += " %s\n" % (["Stopped","Recording"][self.is_recording()])
        status_str  += "  %s frames (in '%s' format) saved to directory '%s' using pattern '%s' .\n" % \
                       (self.getFrameCount(), self.getImgFormat(),self.getImgDir(), self.getInputPattern())
        status_str  += "  Est. movie length is %ss.\n" % (self.getFrameCount()/24)
        status_str  += "------------------------------------------------\n"
        return status_str
                

    def capture_image(self, *_):

        f = self.frame_number + 1 + self.postprocess_frames
        if not self.limit is None and f >= self.limit:
            self.stop_recording()
            return

        self.frame_number = f
        fcount = f + 1
        self._informFrameCount(fcount)
        if fcount % 10 == 0:
            self._notifyStatus("Capturing frame #%d " % fcount)

        save_path = self.image_path(self.frame_number)

        width, height = (None,None) if self.size is None else self.size

        v = self.session.main_view
        rgba = v.image_rgba(width, height, supersample = self.supersample,
                            transparent_background = self.transparent_background)
        color_components = 4 if self.transparent_background else 3
        from PIL import Image
        # Flip y-axis since PIL image has row 0 at top, opengl has row 0 at bottom.
        i = Image.fromarray(rgba[::-1, :, :color_components])
        i.save(save_path, self.img_fmt)
        v.movie_image_rgba = rgba	# Used by crossfade command

        if self.postprocess_frames > 0:
            if self.postprocess_action == 'crossfade':
                self.save_crossfade_images()
            elif self.postprocess_action == 'duplicate':
                self.save_duplicate_images()

    def image_path(self, frame):

        savepat = self.input_pattern.replace('*','%05d')
        basename = savepat % frame
        suffix = '.%s' % self.img_fmt.lower()
        save_filename = basename + suffix
        import os.path
        save_path = os.path.join(self.img_dir, save_filename)
        return save_path

    def save_crossfade_images(self):

        frames = self.postprocess_frames
        self.postprocess_frames = 0
        save_path1 = self.image_path(self.frame_number - frames - 1)
        save_path2 = self.image_path(self.frame_number)
        from os.path import isfile
        if not isfile(save_path1):
            self.stop_recording()
            from chimerax.core.errors import UserError
            raise UserError('movie crossfade cannot be used until one frame has been recorded.')
        from PIL import Image
        image1 = Image.open(save_path1)
        image2 = Image.open(save_path2)
        for f in range(frames):
            imagef = Image.blend(image1, image2, float(f)/(frames-1))
            # TODO: Add save image options as in printer.saveImage()
            pathf = self.image_path(self.frame_number - frames + f)
            imagef.save(pathf, self.img_fmt)
            self._notifyStatus("Cross-fade frame %d " % (f+1))

    def save_duplicate_images(self):

        frames = self.postprocess_frames
        self.postprocess_frames = 0
        save_path = self.image_path(self.frame_number - frames - 1)
        from PIL import Image
        image = Image.open(save_path)
        for f in range(frames):
            pathf = self.image_path(self.frame_number - frames + f)
            image.save(pathf, self.img_fmt)
            self._notifyStatus("Duplicate frame %d " % (f+1))

    def _notifyStatus(self, msg):
        """lower level components (recorder and encoder) call this to
        inform me of some status that needs to be propagated up to the gui
        """
        self.session.logger.status(msg)

    def _notifyError(self, err):
        self.session.logger.info(err)
        self._notifyStatus(err)

    def _notifyInfo(self, info):
        """lower level components (recorder and encoder) call this to
        inform me of some info that needs to be propagated up to the gui
        """
        self.session.logger.info(info)

    def _informEncodingDone(self, exit_val, exit_status, error_msg):
        """ the encoder calls this to notify me that it has finished
        encoding. arguments convey the exit status of the encoding
        process"""

        path = getTruncPath(self.encoder.getOutFile())

        self.encoder = None

        ## You only want to do any kind of reset if the encoder ran
        ## successfully. Don't reset state if encoding was canceled
        ## or there was an error (user may want to change parameters
        ## and re-encode). Also, if resetMode is 'none', you don't want
        ## to do any kind of reset.
        if (exit_status == EXIT_SUCCESS and self.reset_mode != RESET_NONE):
            clr = (self.reset_mode == RESET_CLEAR)
            self.resetRecorder(clearFrames=clr)

        if exit_status == EXIT_SUCCESS:
            success_msg = "Movie saved to %s" % path
            self._notifyStatus(success_msg)
            self._notifyInfo(success_msg + '\n')

        elif  exit_status == EXIT_CANCEL:
            self._notifyStatus("Movie encoding has been canceled.")
            self._notifyInfo("Movie encoding has been canceled.\n")

        elif exit_status == EXIT_ERROR:
            self._notifyError("An error occurred during encoding. See Reply Log for details.")
            self._notifyInfo("\nError during MPEG encoding:\n"
                                "-----------------------------\n"
                                "Exit value:    %s\n"
                                "Error message: \n"
                                "%s\n"
                                "-----------------------------\n" %
                                (exit_val, error_msg)
                                )

    def _informFrameCount(self, count):
        """this method is called by the recorder to inform me of the
        number of frames that have been recorded"""
        if count % 10 == 0:
            self._notifyStatus('Captured %d frames' % count)

    def postprocess(self, action, frames):
        if self.is_recording():
            self.postprocess_action = action
            self.postprocess_frames = frames
            if action == 'duplicate':
                self.capture_image()	# Capture all frames right now.
        else:
            self._notifyStatus('movie %s when not recording' % action)

    def start_encoding(self, output_file, output_format, output_size, video_codec, pixel_format, size_restriction,
                       framerate, bit_rate, quality, round_trip, reset_mode):
        if self.encoder:
            raise MovieError("Currently encoding a movie")

        self.reset_mode = reset_mode

        basepat = self.getInputPattern().replace('*', '%05d')
        suffix = ".%s" % self.getImgFormat().lower()
        pattern = basepat + suffix
        image_dir = self.getImgDir()

        from os.path import join, isfile
        first_image = join(image_dir, pattern % 1)
        if not isfile(first_image):
            raise MovieError("Movie encoding failed because no images were recorded.")

        from .encode import ffmpeg_encoder
        self.encoder = ffmpeg_encoder(output_file, output_format, output_size, video_codec, pixel_format,
                                      size_restriction, framerate, bit_rate, quality, round_trip,
                                      image_dir, pattern, self.getFrameCount(), self._notifyStatus,
                                      self.verbose, self.session)
        self._notifyStatus('Started encoding %d frames' % self.getFrameCount())

        class Status_Reporter:
            def put(self, f):
                f()
        self.encoder.run(Status_Reporter())
        self.encodingFinished()

    def stop_encoding(self):
        if not self.encoder:
            raise MovieError("Not currently encoding")

        self.encoder.abortEncoding()

    # Stop recording when any command error occurs or a script is aborted.
    def haltRecording(self, trigger_name, cdata, tdata):
        if self.is_recording():
            self.resetRecorder(status = False)

    def resetRecorder(self, clearFrames=True, status=True):
        ## when recorder gui calls this, this should tell recordergui what to say
        ## instead of it figureing it out...
        ## so it can also tell command line what to say....

        if self.encoder:
            raise MovieError("Attempted movie reset when encoding not finished.")

        if self.is_recording():
            self.stop_recording()

        if clearFrames:
            self.clearFrames()

        del self.session.movie

        if status:
            msg = ' - frames have been '
            if clearFrames:
                msg += "cleared"
            else:
                import os.path
                ipath = os.path.join(self.getImgDir(), self.getInputPattern())
                msg += "saved (%s.%s)" % (ipath, self.getImgFormat())
            self._notifyStatus("Recorder has been reset %s" % msg)

    def dumpStatusInfo(self):
        self._notifyStatus("Status information written to Reply Log")
        self._notifyInfo(self.getStatusInfo())

    def abortEncoding(self):
        self.encoder.abortEncoding()

    def encodingFinished(self):
        exit_val, exit_status, error_msg = self.encoder.getExitStatus()
            
        if exit_status == EXIT_CANCEL or exit_status == EXIT_ERROR:
            self.encoder.deleteMovie()

        self._informEncodingDone(exit_val, exit_status, error_msg)


def getRandomChars():
    import string, random
    alphanum = string.ascii_letters + string.digits
    return ''.join(random.choice(alphanum) for x in range(4))

def getTruncPath(path):
    path_elts = path.split(os.path.sep)
    ## because save path is absolute, the first elt will be ''
    if not path_elts[0]:
        path_elts = path_elts[1:]
    if len(path_elts) <= 4:
        return path
    else:
        first_two = os.path.join(*path_elts[0:2])
        #print "first_two is ", first_two
        last_two = os.path.join(*path_elts[-2:])
        #print "last_two is ", last_two 
        return os.path.sep + os.path.join(first_two, "...", last_two)
    ## don't need the last 
    #path_elts = path_elts[:-1]
