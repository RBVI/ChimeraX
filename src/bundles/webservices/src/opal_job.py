# vim: set expandtab shiftwidth=4 softtabstop=4:

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

"""
OpalJob - Run Opal job and monitor status
=========================================

OpalJob is a class that runs a web service via
the Opal Toolkit and monitors its status.

Attributes
----------
DEFAULT_OPAL_URL : str
    URL for Opal server to use if none is supplied by caller

"""

from chimerax.core.tasks import Job

# Trailing slash required
DEFAULT_OPAL_URL = "http://webservices.rbvi.ucsf.edu/opal2/services/"


class OpalJob(Job):
    """Launch an Opal Toolkit web service request and monitor its status.

    OpalJob derives from chimerax.core.tasks.Job and
    offers the same API.

    Attributes
    ----------
    service_url : str
        URL to monitor service
    job_id : str
        Opal job id assigned by server
    launch_time : int (seconds since epoch)
        Time when job was launched
    end_time : int (seconds since epoch)
        Time when job terminated

    """
    def __init__(self, *args, **kw):
        """Initialize OpalJob instance.

        Argument
        --------
        logger : chimerax.core.logger.Log instance
            Logger where info and error messages should be sent.

        """
        super().__init__(*args, **kw)
        # Initialize Opal request state
        self.reset_state()

    def start(self, *args, input_file_map=None, **kw):
        # override Job.start so that we can process the input_file_map
        # before start returns, since the files may be temporary
        if input_file_map is None:
            self.input_files = None
        else:
            self.input_files = [self._make_input_file(name, value_type, value)
                           for name, value_type, value in input_file_map]
        super().start(*args, **kw)

    #
    # Define chimerax.core.tasks.Job ABC methods
    #
    def run(self, service_name, cmd, opal_url=None, blocking=False, **kw):
        """Launch the background process.

        Arguments
        ---------
        service_name : str
            Name of Opal service
        cmd : str
            Command line to send to Opal server
        opal_url : str
            URL of Opal server.  If None, DEFAULT_OPAL_URL is used.
        input_file_map : list of tuples
            (actually arg to self.start(), which calls launch())
            List of file names and contents.  Each tuple consists of
            the file name, the type of content, and the content.
            Supported content types include: "text_file", "binary_file"
            and "bytes".  For "text_file" and "binary_file", the value
            should be the path to a file; for "bytes", the value should
            be of type 'bytes'.
        kw : extra arguments
            Arguments passed through to launch request.

        Raises
        ------
        RuntimeError
            If job is already launched
        chimerax.core.tasks.JobLaunchError
            If job failed to launch
        chimerax.core.tasks.JobMonitorError
            If status check failed

        """
        if self.job_id is not None:
            from chimerax.core.tasks import JobError
            raise JobError("Opal job has already been launched")
        logger = self.session.logger
        # TODO: Get proxy host from preferences

        # Get Opal connection
        if opal_url is None:
            opal_url = DEFAULT_OPAL_URL
        self.service_url = opal_url + service_name
        from suds.client import Client
        self._suds = Client(self.service_url + "?wsdl")
        md = self._suds.service.getAppMetadata()
        def _notify(logger=logger, md=md):
            logger.info("Web Service: %s" % md.usage)
        self.session.ui.thread_safe(_notify)

        # Add job keywords
        job_kw = dict([(k[1:], v) for k, v in kw.items() if k.startswith('_')])

        # Create input files if necessary
        if self.input_files is not None:
            job_kw["inputFile"] = self.input_files

        # Launch job
        from suds import WebFault
        import time
        self.launch_time = time.time()
        if not blocking:
            try:
                r = self._suds.service.launchJob(cmd, **job_kw)
            except WebFault as e:
                from chimerax.core.tasks import JobLaunchError
                raise JobLaunchError(str(e))
            else:
                self.job_id = str(r.jobID)
                def _notify(logger=logger, self=self, r=r):
                    logger.info("Opal service URL: %s" % self.service_url)
                    logger.info("Opal job id: %s" % self.job_id)
                    logger.info("Opal status URL prefix: %s" % r.status[2])
                    logger.info("  stdout.txt = standard output")
                    logger.info("  stderr.txt = standard error")
                self.session.ui.thread_safe(_notify)
                self._save_status(r.status)
        else:
            try:
                r = self._suds.service.launchJobBlocking(cmd, **job_kw)
            except WebFault as e:
                from chimerax.core.tasks import JobLaunchError
                raise JobLaunchError(str(e))
            else:
                self.end_time = time.time()
                self._save_status(r.status)
                self._save_outputs(r.jobOut)
        super().run()

    def _save_status(self, status):
        self._status_code = int(status[0])
        self._status_message = str(status[1])
        self._status_url = str(status[2])
        if self._status_code in [4, 8]:
            # 4 == normal finish, 8 == abnormal finish
            import time
            self.end_time = time.time()

    def running(self):
        """Return whether background process is still running.

        """
        return self.launch_time is not None and self.end_time is None

    def monitor(self):
        """Check the status of the background process.

        The task should be marked as terminated in the background
        process is done

        """
        from suds import WebFault
        try:
            r = self._suds.service.queryStatus(self.job_id)
        except WebFault as e:
            from chimerax.core.tasks import JobMonitorError
            raise JobMonitorError(str(e))
        else:
            self._save_status(r)

    def exited_normally(self):
        """Return whether background process terminated normally.

        """
        return self._status_code in [4, 8]

    #
    # Define chimerax.core.session.State ABC methods
    #
    save_attrs = ('service_url', 'job_id', 'launch_time', 'end_time',
                  '_status_code', '_status_message', '_status_url','_outputs')
    def take_snapshot(self, session, flags):
        """Return snapshot of current state of instance.

        The semantics of the data is unknown to the caller.
        Returns None if should be skipped."""
        data = {a:getattr(self,a) for a in self.save_attrs}
        data['version'] = 1
        return data

    @staticmethod
    def restore_snapshot(session, data):
        """Restore data snapshot creating instance."""
        j = OpalJob.__new__(OpalJob)
        for a in self.save_attrs:
            if a in data:
                setattr(j, a, data[a])
        return j

    def reset_state(self):
        """Reset state to data-less state"""
        for a in self.save_attrs:
            setattr(self, a, None)

    #
    # Other Opal-specific methods
    #
    def get_file(self, filename, *, encoding='utf-8'):
        """Return the contents of file on Opal server.

        Argument
        --------
        filename : str
            Name of file on Opal server.
        encoding : str or None
            Encoding to use to decode the file contents (default: utf-8).
            If None, return the raw bytes object.

        Returns
        -------
        str/bytes
            Contents of file.  See 'encoding' argument.

        Raises
        ------
        KeyError
            If file does not exist on server.

        """
        if self._outputs is None:
            if self.end_time is None:
                if filename in ("stdout.txt", "stderr.txt"):
                    url = self._status_url + '/' + filename
                else:
                    raise ValueError("Cannot fetch non-stdout/stderr files before job finishes")
            else:
                self.get_outputs()
                url = self._outputs[filename]
        else:
            url = self._outputs[filename]
        from urllib.request import urlopen
        with urlopen(url) as f:
            contents = f.read()
        if encoding is None:
            return contents
        return contents.decode(encoding)

    def get_stdout(self):
        return self.get_file("stdout.txt")

    def get_stderr(self):
        return self.get_file("stderr.txt")

    def get_outputs(self, refresh=False):
        """Return dictionary of output files and their URLs.

        This method need not be called explicitly if the
        names of output files are known.  It will be called
        automatically to find the corresponding URL by
        'get_file'.

        Returns
        dict
            Dictionary whose keys are file names and values
            are corresponding URLs.

        """
        if not refresh and self._outputs is not None:
            return self._outputs
        from suds import WebFault
        try:
            r = self._suds.service.getOutputs(self.job_id)
        except WebFault as e:
            from chimerax.core.tasks import JobError
            raise JobError(str(e))
        self._save_outputs(r)

    def _save_outputs(self, r):
        self._outputs = {
            "stdout.txt": r.stdOut,
            "stderr.txt": r.stdErr,
        }
        try:
            files = r.outputFile
        except AttributeError:
            pass
        else:
            self._outputs.update([(f.name, f.url) for f in files])
        return self._outputs

    def _make_input_file(self, name, value_type, value):
        # text files are opened normally, with contents encoded as UTF-8.
        # binary files are opened in binary mode and untouched.
        # bytes are used as is.
        if value_type == "text_file":
            with open(value, "r") as f:
                value = f.read().encode("UTF-8")
        elif value_type == "binary_file":
            with open(value, "rb") as f:
                value = f.read()
        elif value_type != "bytes":
            raise ValueError("unsupported content type: \"%s\"" % value_type)
        # Opal wants base64 encoded content as string
        from base64 import b64encode
        contents = b64encode(value).decode("UTF-8")
        return {"name": name, "contents": contents}
