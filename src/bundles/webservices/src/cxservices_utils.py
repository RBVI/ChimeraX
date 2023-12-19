from cxservices.api import default_api
from cxservices.rest import ApiException

def get_status(job_id):
    """Check the status of the background process.

    The task should be marked as terminated in the background
    process is done
    """
    api = default_api.DefaultApi()
    try:
        status = api.status(job_id).status
    except ApiException as e:
        raise JobMonitorError(str(e))
    return status

def get_results(job_id):
    api = default_api.DefaultApi()
    try:
        content = api.get_results(job_id)
    except ApiException:
        return None
    else:
        return content

def job_exited_normally(job_id):
    """Return whether background process terminated normally.

    """
    return get_status(job_id) == "complete"

def get_file(job_id, filename, *, encoding='utf-8'):
    api = default_api.DefaultApi()
    try:
        content = api.file_get(job_id, filename)
    except ApiException as e:
        raise KeyError("%s: %s" % (filename, str(e)))
    if encoding is None:
        return content
    else:
        return content.decode(encoding)

def get_stdout(job_id):
    return get_file(job_id, "_stdout")

def get_stderr(job_id):
    return get_file(job_id, "_stdout")
