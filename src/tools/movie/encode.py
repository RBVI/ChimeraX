class ffmpeg_encoder:

    def __init__(self,
                 output_file,
                 output_format,
                 output_size,
                 video_codec,
                 pixel_format,
                 size_restriction,
                 framerate,
                 bit_rate,
                 quality,
                 round_trip,
                 image_directory,
                 image_file_pattern,
                 image_count,
                 status = None,
                 session = None,
                 ffmpeg_cmd = 'ffmpeg'):

        self.session = session
        self.image_directory = image_directory
        self.image_file_pattern = image_file_pattern
        self.image_count = image_count
        self.status = status
        import os.path
        self.output_file = os.path.expanduser(output_file)
        self.ffmpeg_cmd = ffmpeg_cmd

        self.arg_list = self._buildArgList(output_file, output_format, output_size, video_codec, pixel_format,
                                           size_restriction, framerate, bit_rate, quality, session)

        if round_trip:
            self.copy_frames_backwards()

        import threading
        self.encodeAbortEvt = threading.Event()

        self.exit_status = (None, None, None)

    def _buildArgList(self, output_file, output_format, output_size, video_codec, pixel_format,
                      size_restriction, framerate, bit_rate, quality, session):

        arg_list = []

        # Frame rate command-line option must appear before input files
        # command-line option because the rate describes the input sequence.
        # Putting the -r after the input files has the undesired effect of
        # eliminating frames at rates lower than 25 fps to keep the duration
        # equal to the 25 fps input sequence duration.
        arg_list.append('-r')
        arg_list.append(str(framerate))

        arg_list.append('-i')
        import os.path
        arg_list.append (os.path.join(self.image_directory, self.image_file_pattern))

        r = size_restriction
        if not r is None:
            arg_list.append('-vf')
            wd,hd = r
            arg_list.append('crop=floor(in_w/%d)*%d:floor(in_h/%d)*%d:0:0' % (wd,wd,hd,hd))

        arg_list.append('-y')        # overwrite the output file

        arg_list.append('-vcodec')
        arg_list.append(video_codec)

        arg_list.append('-f')
        arg_list.append(output_format)

        arg_list.append('-pix_fmt')
        arg_list.append(pixel_format)

        if output_size:
            arg_list.append('-s')
            arg_list.append(output_size)

        if bit_rate:
            arg_list.append('-vb')
            arg_list.append('%dk' % bit_rate)

        if quality:
            qopt, qval = quality
            arg_list.append(qopt)
            arg_list.append(str(qval))

        path = output_file
        from os.path import dirname, isdir, join
        d = dirname(path)
        if d == '':
            path = os.path.join(os.getcwd(), path)
        elif not isdir(d):
            from .movie import MovieError
            raise MovieError('Output directory does not exist: %s' % d)
        arg_list.append(path)

        ffmpeg_exe = join(session.app_bin_dir, self.ffmpeg_cmd)
        arg_list.insert(0, ffmpeg_exe)
        session.logger.info(' '.join(arg_list))

        return arg_list

    def copy_frames_backwards(self):

        import os.path
        pat = os.path.join(self.image_directory, self.image_file_pattern)
        n = self.image_count + 1
        self.loop = (pat, n)
        import os
        if hasattr(os, 'link'):
            copy = os.link
        else:
            from shutil import copyfile
            copy = copyfile      # Cannot link on Windows XP
        for f in range(n):
            pto = pat % (n+f)
            if os.path.exists(pto):
                os.remove(pto)
            copy(pat % (n-1-f), pto)
            
    def remove_backwards_frames(self):

        if hasattr(self, 'loop'):
            import os
            pat, n = self.loop
            for f in range(n):
                os.remove(pat % (n+f))
            
    def _parseOutput(self, out_line):
        ## frame=   36 q=2.0 Lsize=      28kB time=1.4 bitrate= 163.8kbits/s
        if out_line[0:6] == "frame=":
            return out_line
        else:
            return None

    def parseOutput(self, out_line):
        #ESTIMATED TIME OF COMPLETION:  0 seconds
        #FRAME 35 (B):  I BLOCKS:  0;  B BLOCKS:  685   SKIPPED:  251 (0 seconds)

        if out_line.find('ESTIMATED TIME OF COMPLETION') >= 0:
            time_info = out_line.split(':',1)[1].strip()
            amt, units = time_info.split()
            if units.strip() == 'minutes':
                return {'time': '%s min.' % amt}
            else:
                return {'time': '%s sec.' % self.convertToMinSecs(amt)}

        elif out_line[0:5] == 'FRAME':
            frame_info = out_line.split(':',1)[0]
            frame_num = frame_info.split()[1]
            return {'frame':frame_num}
        else:
            return None

    def getOutFile(self):
        return self.output_file

    def abortEncoding(self):
        self.encodeAbortEvt.set()

    def deleteMovie(self):
        import os
        try:
            os.remove(self.output_file)
        except:
            pass

    def killEncoder(self, pop_obj):
        try:
                pop_obj.stdin.write('q')
                pop_obj.stdin.flush()
        except IOError:
                pass

    def convertToMinSecs(self, secs):
        secs = int(secs)

        if secs < 60:
            return "%s" % secs
        else:
            return "%d:%02d" % (secs/60, secs%60)

    def getExitStatus(self):
        return self.exit_status

    def run(self, message_queue):

        from .movie import EXIT_SUCCESS, EXIT_ERROR, EXIT_CANCEL

        arg0 = self.arg_list[0]
        import os.path
        if not os.path.isfile(arg0):
            self.exit_status = (-1, EXIT_ERROR,
                                 'Could not find %s executable at %s' % (self.ffmpeg_cmd, arg0))
            return

        # all output is on stderr, but Windows needs all standard I/O to
        # be redirected if one is, so stdout is a pipe too
        from io import StringIO
        out = StringIO()
        from subprocess import Popen, PIPE
        p = Popen(self.arg_list, stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        exit_code = p.returncode

        status = EXIT_SUCCESS if exit_code == 0 else EXIT_ERROR
        ffmpeg_output = 'stdout:\n%s\nstderr:\n%s' % (out.decode('utf-8'), err.decode('utf-8'))
        error = '' if exit_code == 0 else ffmpeg_output
        self.exit_status = (exit_code, status, error)

        self.session.logger.info(ffmpeg_output)

        self.remove_backwards_frames()
