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
CxServicesJob - Run ChimeraX REST job and monitor status
=========================================

CxServicesJob is a class that runs a web service via
the ChimeraX REST server and monitors its status.
"""
import json
import time
import logging
from urllib3.exceptions import MaxRetryError, NewConnectionError
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from urllib.error import URLError

from chimerax.core.tasks import Job, JobError, JobLaunchError, JobMonitorError

from cxservices.rest import ApiException
from cxservices.api import default_api

class CxServicesJob(Job):
    """Launch a ChimeraX REST web service request and monitor its status.

    CxServicesJob derives from chimerax.core.tasks.Job and
    offers the same API.

    Attributes
    ----------
    job_id : str
        ChimeraX REST job id assigned by server
    launch_time : int (seconds since epoch)
        Time when job was launched
    end_time : int (seconds since epoch)
        Time when job terminated
    status : str
    outputs : List[str]
    next_poll : time
    """
    save_attrs = ('job_id', 'launch_time', 'end_time', 'status', 'outputs', 'next_poll')
    chimerax_api = default_api.DefaultApi()
    # Ticket #6187, set urllib3 not to log messages to the general ChimeraX log
    logging.getLogger("urllib3").setLevel(100)

    def reset_state(self) -> None:
        """Reset state to data-less state"""
        for a in self.save_attrs:
            setattr(self, a, None)

    def __init__(self, *args, **kw) -> None:
        """Initialize CxServicesJob instance.

        Argument
        --------
        logger : chimerax.core.logger.Log instance
            Logger where info and error messages should be sent.

        """
        super().__init__(*args, **kw)
        # Initialize ChimeraX REST request state
        self.reset_state()

    @property
    def status(self) -> str:
        return self._status

    @status.setter
    def status(self, value) -> None:
        self._status = value

    def run(self, service_name: str
            , params: Dict[str, Any] = None
            , files_to_upload: Optional[List[str]] = None) -> None:
        """Launch the background process.

        Arguments
        ---------
        service_name: Name of REST service
        params: Dictionary of parameters to send to REST server
        files_to_upload: Dictionary of files to upload to REST server

        Raises
        ------
        RuntimeError
            If job is already launched
        chimerax.core.tasks.JobLaunchError
            If job failed to launch
        chimerax.core.tasks.JobMonitorError
            If status check failed
        """
        # We have to do this so that urrllib3, which swagger's generated
        # API calls, can serialize the params dict.
        processed_params = json.dumps(params)
        processed_files_to_upload = None
        if files_to_upload is not None:
            processed_files_to_upload = {"job_files": files_to_upload}
        if self.launch_time is not None:
            raise JobError("REST job has already been launched")
        self.launch_time = time.time()
        # Launch job
        try:
            result = self.chimerax_api.submit_job(
                job_type = service_name
                , params = processed_params
                , filepaths = processed_files_to_upload
            )
        except ApiException as e:
            self.status = "failed"
            self.end_time = time.time()
            raise JobLaunchError(str(e))
        except (URLError, MaxRetryError, NewConnectionError) as e:
            self.status = "failed"
            self.end_time = time.time()
            self.thread_safe_log(
                "Error launching job: ChimeraX Web Services unavailable. Please try again soon."
            )
        else:
            self.job_id = result.job_id
            self.urls = {
                "status": result.status_url,
                "results": result.results_url,
            }
            self.next_poll = int(result.next_poll)
            self.thread_safe_log("Webservices job id: %s" % self.job_id)
            super().run()

    def running(self) -> bool:
        """Return whether background process is still running.

        """
        return self.launch_time is not None and self.end_time is None

    def next_check(self) -> Optional[int]:
        return self.next_poll

    def monitor(self, poll_freq_override: Optional[int] = None) -> None:
        """Check the status of the background process.

        The task should be marked as terminated in the background
        process is done
        """
        try:
            # Not sure why, but we have to specify job_id by name here
            result = self.chimerax_api.get_status(job_id = self.job_id)
            status = result.status
            next_poll = result.next_poll
        except ApiException as e:
            raise JobMonitorError(str(e))
        self.status = status
        if poll_freq_override is None and next_poll is not None:
            self.next_poll = int(next_poll)
        else:
            self.next_poll = poll_freq_override
        if status in ["finished","failed","deleted","canceled"] and self.end_time is None:
            self.end_time = time.time()

    def exited_normally(self) -> bool:
        """Return whether background process terminated normally.

        """
        return self._status == "finished"

    def get_results(self) -> Optional[Union[bytes,str]]:
        """Expects JSON."""
        try:
            content = self.chimerax_api.get_results(self.job_id)
        except ApiException:
            return None
        else:
            return content

    #
    # Other helper methods
    #
    def get_file(self, filename, *, encoding='utf-8'):
        """Return the contents of file on REST server.

        Argument
        --------
        filename : str
            Name of file on REST server.
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
        try:
            content = self.chimerax_api.get_file(self.job_id, filename)
        except ApiException as e:
            raise KeyError("%s: %s" % (filename, str(e)))
        if encoding is None:
            return content
        else:
            return content.decode(encoding)

    def get_stdout(self):
        return self.get_file("_stdout")

    def get_stderr(self):
        return self.get_file("_stdout")

    def get_all_filenames(self, refresh=False):
        """Return dictionary of output files and their URLs.

        This method need not be called explicitly if the
        names of output files are known.  It will be called
        automatically to find the corresponding URL by
        'get_file'.

        Returns
        dict
            Dictionary whose keys are file names and values
            are appropriate for use with get_file().

        """
        if not refresh and self._outputs is not None:
            return self._outputs
        try:
            filenames = self.chimerax_api.get_job_filenames(self.job_id).files
        except ApiException as e:
            raise IOError("job %s: cannot get file list" % self.job_id)
        self._outputs = {fn:fn for fn in filenames}
        return self._outputs

    def post_file(self, name, value_type, value) -> None:
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
        self.chimerax_api.file_post(value, self.job_id, name)

    def __str__(self) -> str:
        return "CxServicesJob (ID: %s)" % self.id

    @classmethod
    def from_snapshot(cls, session, data):
        tmp = cls()
        if data.version == 1:
            for a in self.save_attrs:
                if a in data:
                    setattr(tmp, a, data[a])
            if tmp.end_time is None:
                tmp.chimerax_api = default_api.DefaultApi()
        else:
            for a in self.save_attrs:
                if a in data:
                    setattr(tmp, a, data[a])
            if tmp.end_time is None:
                tmp.chimerax_api = default_api.DefaultApi()
        return tmp

    #
    # Define chimerax.core.session.State ABC methods
    #
    def take_snapshot(self, session, flags) -> Dict:
        """Return snapshot of current state of instance.

        The semantics of the data is unknown to the caller.
        Returns None if should be skipped."""
        data = {a:getattr(self,a) for a in self.save_attrs}
        data['version'] = 2
        return data

    @staticmethod
    def restore_snapshot(session, data) -> 'CxServicesJob':
        """Restore data snapshot creating instance."""
        return CxServicesJob.from_snapshot(session, data)
