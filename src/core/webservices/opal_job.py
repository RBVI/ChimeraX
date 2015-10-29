# vim: set expandtab shiftwidth=4 softtabstop=4:
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

from ..tasks import Job

# Trailing slash required
DEFAULT_OPAL_URL = "http://webservices.rbvi.ucsf.edu/opal2/services/"


class OpalJob(Job):
    """Launch an Opal Toolkit web service request and monitor its status.

    OpalJob derives from chimera.core.tasks.Job and
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
        logger : chimera.core.logger.Log instance
            Logger where info and error messages should be sent.

        """
        super().__init__(*args, **kw)
        # Initialize Opal request state
        self.reset_state()

    #
    # Define chimera.core.tasks.Job ABC methods
    #
    def launch(self, service_name, cmd, opal_url=None,
               input_file_map=None, **kw):
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
        chimera.core.tasks.JobLaunchError
            If job failed to launch
        chimera.core.tasks.JobMonitorError
            If status check failed

        """
        if self.job_id is not None:
            from ..tasks import JobError
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
        logger.info("Web Service: %s" % md.usage)

        # Add job keywords
        job_kw = dict([(k[1:], v) for k, v in kw.items() if k.startswith('_')])

        # Create input file map if necessary
        if input_file_map is not None:
            input_files = [self._make_input_file(name, value_type, value)
                           for name, value_type, value in input_file_map]
            job_kw["inputFile"] = input_files

        # Launch job
        from suds import WebFault
        try:
            r = self._suds.service.launchJob(cmd, **job_kw)
        except WebFault as e:
            from ..tasks import JobLaunchError
            raise JobLaunchError(str(e))
        else:
            self.job_id = r.jobID
            logger.info("Opal service URL: %s" % self.service_url)
            logger.info("Opal job id: %s" % self.job_id)
            logger.info("Opal status URL: %s" % r.status[2])
            import time
            self.start_time = time.time()
            self._save_status(r.status)

    def _save_status(self, status):
        self._status_code = int(status[0])
        print("_status_code", self._status_code)
        self._status_message = str(status[1])
        self._status_url = str(status[2])
        if self._status_code in [4, 8]:
            # 4 == normal finish, 8 == abnormal finish
            import time
            self.end_time = time.time()

    def running(self):
        """Return whether background process is still running.

        """
        return self.start_time is not None and self.end_time is None

    def monitor(self):
        """Check the status of the background process.

        The task should be marked as terminated in the background
        process is done

        """
        from suds import WebFault
        try:
            r = self._suds.service.queryStatus(self.job_id)
        except WebFault as e:
            from ..tasks import JobMonitorError
            raise JobMonitorError(str(e))
        else:
            self._save_status(r)

    def exited_normally(self):
        """Return whether background process terminated normally.

        """
        return self._status_code in [4, 8]

    #
    # Define chimera.core.session.State ABC methods
    #
    def take_snapshot(self, phase, session, flags):
        """Return snapshot of current state, [version, data], of instance.

        The semantics of the data is unknown to the caller.
        Returns None if should be skipped."""
        if phase != self.SAVE_PHASE:
            return
        version = 1
        data = [
            self.service_url,
            self.job_id,
            self.launch_time,
            self.end_time,
            self._status_code,
            self._status_message,
            self._status_url,
            self._outputs,
        ]
        return [version, data]

    def restore_snapshot(self, phase, session, version, data):
        """Restore data snapshot into instance.

        Restoration is done in two phases: CREATE_PHASE and RESOLVE_PHASE.  The
        first phase should restore all of the data.  The
        second phase should restore references to other objects (data is None).
        The session instance is used to convert unique ids into instances.
        """
        if phase != self.CREATE_PHASE:
            return
        if version == 1:
            self.service_url = data[0]
            self.job_id = data[1]
            self.launch_time = data[2]
            self.end_time = data[3]
            self._status_code = data[4]
            self._status_message = data[5]
            self._status_url = data[6]
            self._outputs = data[7]
            return
        from ..session import RestoreError
        raise RestoreError("Unexpected version")

    def reset_state(self):
        """Reset state to data-less state"""
        self.service_url = None
        self.job_id = None
        self.launch_time = None
        self.end_time = None
        self._status_code = None
        self._status_message = None
        self._status_url = None
        self._outputs = None

    #
    # Other Opal-specific methods
    #
    def get_file(self, filename):
        """Return the contents of file on Opal server.

        Argument
        --------
        filename : str
            Name of file on Opal server.

        Returns
        -------
        str
            Contents of file.

        Raises
        ------
        KeyError
            If file does not exist on server.

        """
        if self._outputs is None:
            self.get_outputs()
        from urllib.request import urlopen
        with urlopen(self._outputs[filename]) as f:
            return f.read()

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
            from ..tasks import JobError
            raise JobError(str(e))
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
