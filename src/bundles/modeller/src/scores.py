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

def fetch_scores(session, structures, *, block=True, license_key=None, refresh=False):
    """
    Fetch scores for models from Modeller web site.

    Arguments:
    session
        current session
    structures
        Structures that should be scored.  Attributes of the form modeller_XXX will be
        assigned to them, where XXX is the score name.
    block
        If True, wait for all scoring jobs to finish before returning.  Otherwise return immediately.
    license_key
        Modeller license key.  If not provided, try to use settings to find one.
    refresh
        Normally existing scores of a particular type are more accurate than those fetched from the
        web, and are therefore retained (refresh == False).  If 'refresh' is True, existing scores
        will be overwitten by fetched scores.
    """

    from .common import get_license_key
    license_key = get_license_key(session, license_key)

    # the Modeller scoring server throttles job execution on its own, so we can launch
    # all the scoring jobs at once without crippling the server
    from chimerax.webservices.httpq import HTTPQueue
    httpq = HTTPQueue(session, thread_max=None)

    session.logger.status("Initiating %d scoring requests to Modeller evaluation"
        " server" % len(structures))

    modeller_host = "modbase.compbio.ucsf.edu"
    from threading import Lock
    results_lock = Lock()
    for s in structures:
        slot = httpq.new_slot(modeller_host)
        slot.request(ModellerScoringJob, session, modeller_host, license_key, refresh, s)

    if block:
        # The jobs update score in their "on_finish" method, which runs in the main thread,
        # so we need to process events in order to allow that to happen before returning.
        # This doesn't allow them to update as they _individually_ finish, but that doesn't
        # seem to really be relevant to a blocking execution scenario
        httpq.wait()
        if session.ui.is_gui:
            session.ui.processEvents()

from chimerax.core.tasks import Job
class ModellerScoringJob(Job):

    def __init__(self, session, modeller_host, license_key, refresh, structure):
        super().__init__(session)
        self.start(session, modeller_host, license_key, structure, refresh, blocking=True)

    def launch(self, session, modeller_host, license_key, structure, refresh, **kw):
        self.structure = structure
        self.results = {'chimerax refresh': refresh}

        thread_safe = session.ui.thread_safe
        from io import StringIO
        from chimerax.pdb import save_pdb
        pdb_buffer = StringIO()
        save_pdb(session, pdb_buffer, models=[structure])
        fields = [
            ("name", None, "chimerax-Modeller_Comparative"),
            ("modkey", None, license_key),
            # Sequence identity info will be taken from PDB header for Modeller models,
            # and since we don't know the info otherwise -- skip it
            ("model_file", structure.name.replace(' ', '_') + '.pdb', pdb_buffer.getvalue())
        ]
        from chimerax.webservices.post_form import post_multipart
        submission = post_multipart(modeller_host, "/modeval/job", fields, ssl=True,
            accept_type="application/xml")
        from xml.dom.minidom import parseString
        sub_dom = parseString(submission)
        top = sub_dom.getElementsByTagName('saliweb')[0]
        for results in top.getElementsByTagName('job'):
            url = results.getAttribute("xlink:href")
            sub_dom.unlink()
            break
        else:
            sub_dom.unlink()
            thread_safe(session.logger.error, "Cannot submit evaluation job for %s" % structure.name)
            return

        # wait for scoring job to finish...
        from urllib.request import urlopen
        from urllib.error import URLError, HTTPError
        while True:
            try:
                output = urlopen(url)
                break
            except (URLError, HTTPError) as e:
                if e.code == 503:
                    import time
                    time.sleep(5)
                else:
                    thread_safe(session.logger.error, "Cannot fetch scoring results for %s: %s"
                        % (structure.name, e))
                    return

        # parse output
        out_dom = parseString(output.read())
        top = out_dom.getElementsByTagName('saliweb')[0]
        for results in top.getElementsByTagName('results_file'):
            results_url = results.getAttribute('xlink:href')
            if "evaluation.xml" in results_url:
                eval_out = urlopen(results_url)
                eval_dom = parseString(eval_out.read())
                from chimerax.core.utils import string_to_attr
                for name in ["zDOPE", "predicted_RMSD", "predicted_NO35"]:
                    structure.__class__.register_attr(session, string_to_attr(name, prefix="modeller_"),
                        "Modeller", attr_type=float)
                    val = float(eval_dom.getElementsByTagName(name.lower())[0].firstChild.nodeValue.strip())
                    self.results[name] = val

    def running(self):
        return len(self.results) < 2

    def on_finish(self):
        if len(self.results) > 1:
            if not self.structure.deleted:
                refresh = self.results.pop('chimerax refresh')
                for attr_suffix, val in self.results.items():
                    attr_name = "modeller_" + attr_suffix
                    if refresh or not hasattr(self.structure, attr_name):
                        setattr(self.structure, attr_name, val)
                        self.session.change_tracker.add_modified(self.structure, attr_name + " changed")
                self.session.logger.status("Modeller scores for %s fetched" % self.structure)
