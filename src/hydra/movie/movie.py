EXIT_ERROR = "ERROR"
EXIT_CANCEL = "CANCEL"
EXIT_SUCCESS = "SUCCESS"

DEFAULT_PATTERN = "chimovie_%s-*"
import os.path
DEFAULT_OUTFILE = os.path.expanduser("~/Desktop/movie.mp4")

RESET_CLEAR = 'clear'
RESET_KEEP = 'keep'
RESET_NONE = 'none'

from ..ui.commands import CommandError
MovieError = CommandError

class Movie:

    def __init__(self, img_fmt=None, img_dir=None, input_pattern=None,
                 size=None, supersample=0, limit = None, session = None):

        self.session = session

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
        self.limit = limit

        self.newFrameHandle = None

        self.frame_count = -1

        self.postprocess_action = None
        self.postprocess_frames = 0

        self.recording = False
        self.task = None
        self.encoder = None
        self.encoding_thread = None
        self.resetMode = None

#        from chimera import triggers, COMMAND_ERROR, SCRIPT_ABORT
#        triggers.addHandler(COMMAND_ERROR, self.haltRecording, None)
#        triggers.addHandler(SCRIPT_ABORT, self.haltRecording, None)

    def start_recording(self):
        v = self.session.view
        v.add_rendered_frame_callback(self.capture_image)
        self.recording = True
#        from chimera.tasks import Task
#        self.task = Task("record movie", self.cancelCB)
        
    def is_recording(self):
        return self.recording

    def postprocess(self, action, frames):
        if self.frame_count < 0:
            # Need an initial image to do a crossfade.
            self.capture_image()
        self.postprocess_action = action
        self.postprocess_frames = frames

    def cancelCB(self):
        # If user cancels inside of capture_image, this callback
        # is never invoked because of the exception thrown in saveImage
        # happens before this function is called.  So we 
        self.stop_recording()
        self._notifyStatus("movie recording aborted by user")

    def stop_recording(self):
        if not self.is_recording():
            raise MovieError("Not currently recording")
        v = self.session.view
        v.remove_rendered_frame_callback(self.capture_image)
        self.recording = False
#        self.task.finished()
        self.task = None

    def reset(self):
        self.frame_count = -1
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
            except:
                self.session.show_info("Error removing file %s" % s)


    def getFrameCount(self):
        return self.frame_count

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
                

    def capture_image(self):

        f = self.frame_count + 1 + self.postprocess_frames
        if not self.limit is None and f >= self.limit:
            self.stop_recording()
            return

        self.frame_count += 1 + self.postprocess_frames
        self._informFrameCount(self.frame_count)
        if self.frame_count%10 == 0:
            self._notifyStatus("Capturing frame #%d " % self.frame_count)

        save_path = self.image_path(self.frame_count)

        width, height = (None,None) if self.size is None else self.size

        v = self.session.view
        from ..file_io.opensave import save_image
        save_image(save_path, self.session, width, height, self.img_fmt)

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
        save_path1 = self.image_path(self.frame_count - frames - 1)
        save_path2 = self.image_path(self.frame_count)
        from PIL import Image
        image1 = Image.open(save_path1)
        image2 = Image.open(save_path2)
        for f in range(frames):
            imagef = Image.blend(image1, image2, float(f)/(frames-1))
            # TODO: Add save image options as in printer.saveImage()
            pathf = self.image_path(self.frame_count - frames + f)
            imagef.save(pathf, self.img_fmt)
            self._notifyStatus("Cross-fade frame %d " % (f+1))

    def save_duplicate_images(self):

        frames = self.postprocess_frames
        self.postprocess_frames = 0
        save_path = self.image_path(self.frame_count - frames - 1)
        from PIL import Image
        image = Image.open(save_path)
        for f in range(frames):
            pathf = self.image_path(self.frame_count - frames + f)
            image.save(pathf, self.img_fmt)
            self._notifyStatus("Duplicate frame %d " % (f+1))

    def _notifyStatus(self, msg):
        """lower level components (recorder and encoder) call this to
        inform me of some status that needs to be propagated up to the gui
        """
        self.session.show_status(msg)

    def _notifyError(self, err):
        self.session.show_info(err)
        self._notifyStatus(err)

    def _notifyInfo(self, info):
        """lower level components (recorder and encoder) call this to
        inform me of some info that needs to be propagated up to the gui
        """
        self.session.show_info(info)

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
            self.postprocess(action, frames)
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

        from .encode import ffmpeg_encoder
        self.encoder = ffmpeg_encoder(output_file, output_format, output_size, video_codec, pixel_format,
                                      size_restriction, framerate, bit_rate, quality, round_trip,
                                      self.getImgDir(), pattern, self.getFrameCount(), self._notifyStatus,
                                      self.session)
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

        self.session.movie = None

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
