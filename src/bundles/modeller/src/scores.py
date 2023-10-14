# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===
from chimerax.core.tasks import Job

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
    # Was previously results_lock
    _ = Lock()
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


class ModellerScoringJob(Job):

    def __init__(self, session, modeller_host, license_key, refresh, structure):
        super().__init__(session)
        self.start(session, modeller_host, license_key, structure, refresh, blocking=True)

    def run(self, session, modeller_host, license_key, structure, refresh, **kw):
        self.structure = structure
        self.results = {'chimerax refresh': refresh}

        def threaded_run(self=self, session=session, modeller_host=modeller_host, license_key=license_key):
            thread_safe = session.ui.thread_safe
            from io import StringIO
            from chimerax.pdb import save_pdb
            pdb_buffer = StringIO()
            save_pdb(session, pdb_buffer, models=[self.structure])
            fields = [
                ("name", None, "chimerax-Modeller_Comparative"),
                ("modkey", None, license_key),
                # Sequence identity info will be taken from PDB header for Modeller models,
                # and since we don't know the info otherwise -- skip it
                ("model_file", self.structure.name.replace(' ', '_') + '.pdb', pdb_buffer.getvalue())
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
                thread_safe(session.logger.error, "Cannot submit evaluation job for %s"
                    % self.structure.name)
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
                                    % (self.structure.name, e))
                        return

            # parse output
            out_dom = parseString(output.read())
            top = out_dom.getElementsByTagName('saliweb')[0]
            for results in top.getElementsByTagName('results_file'):
                results_url = results.getAttribute('xlink:href')
                if "evaluation.xml" in results_url:
                    eval_out = urlopen(results_url)
                    eval_dom = parseString(eval_out.read())
                    from chimerax.core.attributes import string_to_attr
                    for name in ["zDOPE", "predicted_RMSD", "predicted_NO35"]:
                        self.structure.__class__.register_attr(session, string_to_attr(name,
                            prefix="modeller_"), "Modeller", attr_type=float)
                        val = float(eval_dom.getElementsByTagName(
                            name.lower())[0].firstChild.nodeValue.strip())
                        self.results[name] = val
        import threading
        thread = threading.Thread(target=threaded_run, daemon=True)
        thread.start()
        super().run()

    def monitor(self):
        session = self.structure.session
        session.ui.thread_safe(session.logger.status, "Modeller scoring job for %s still running"
            % self.structure)

    def next_check(self):
        import random
        return random.randint(5, 20)

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
