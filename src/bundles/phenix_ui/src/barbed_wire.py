# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

#
# Command to place a ligand in a cryoEM map using Phenix ligandfit.
#
from chimerax.core.tasks import Job
from chimerax.core.errors import UserError
from chimerax.atomic import AtomicStructure, Atom, colors, Residue
from time import time

class BarbedWireJob(Job):

    SESSION_SAVE = False

    def __init__(self, session, executable_location, optional_args, positional_args, temp_dir, verbose,
            callback, block):
        super().__init__(session)
        self._running = False
        self._monitor_time = 0
        self._monitor_interval = 10
        self.start(session, executable_location, optional_args, positional_args, temp_dir, verbose,
            callback, blocking=block)

    def run(self, session, executable_location, optional_args, positional_args, temp_dir, verbose,
            callback, **kw):
        self._running = True
        self.start_t = time()
        def threaded_run(self=self):
            try:
                results = _run_barbed_wire_subprocess(session, executable_location, optional_args,
                    positional_args, temp_dir, verbose)
            except Exception as e:
                from .util import thread_throw
                thread_throw(session, e)
                return
            finally:
                self._running = False
            self.session.ui.thread_safe(callback, results)
        import threading
        thread = threading.Thread(target=threaded_run, daemon=True)
        thread.start()
        super().run()

    def monitor(self):
        from chimerax.core.commands import plural_form
        plural_seconds = lambda n: plural_form(n, "second")
        plural_minutes = lambda n: plural_form(n, "minute")
        delta = int(time() - self.start_t + 0.5)
        if delta < 60:
            time_info = "%d %s" % (delta, plural_seconds(delta))
        elif delta < 3600:
            minutes = delta // 60
            seconds = delta % 60
            time_info = "%d %s and %d %s" % (minutes, plural_minutes(minutes), seconds,
                plural_seconds(seconds))
        else:
            hours = delta // 3600
            minutes = (delta % 3600) // 60
            seconds = delta % 60
            time_info = "%d:%02d:%02d" % (hours, minutes, seconds)
        ses = self.session
        ses.ui.thread_safe(ses.logger.status, "Barbed wire job still running (%s)" % time_info)

    def next_check(self):
        return self._monitor_interval
        self._monitor_time += self._monitor_interval
        return self._monitor_time

    def running(self):
        return self._running

def phenix_barbed_wire(session, structures, *, block=None, phenix_location=None, key=True,
        verbose=False, option_arg=[], position_arg=[]):

    # Find the phenix.barbed_wire_analysis executable
    from .locate import find_phenix_command
    exe_path = find_phenix_command(session, 'phenix.barbed_wire_analysis', phenix_location)

    # if blocking not explicitly specified, block if in a script or in nogui mode
    if block is None:
        block = session.in_script or not session.ui.is_gui

    if structures is None:
        from chimerax.atomic import all_atomic_structures
        structures = all_atomic_structures(session)
        if not structures:
            raise UserError("No structures currently open")

    # Setup temporary directory to run phenix.barbed_wire_analysis
    from tempfile import TemporaryDirectory
    tdir = TemporaryDirectory(prefix = 'barbed_wire_analysis_')  # Will be cleaned up when object deleted.
    temp_dir = tdir.name

    for s in structures:
        # Save model to file.
        from chimerax.pdb import save_pdb
        from os import path
        pdb_location = path.join(temp_dir,'model.pdb')
        save_pdb(session, pdb_location, models=[s])

        # Run phenix.barbed_wire_analysis
        # keep a reference to 'tdir' in the callback so that the temporary directory isn't removed before
        # the program runs
        callback = lambda json, *args, session=session, model=s, show_key=key, d_ref=tdir: \
            _process_results(session, json, model, show_key)
        BarbedWireJob(session, exe_path, option_arg, position_arg, temp_dir, verbose, callback, block)

def _process_results(session, json, structure, show_key):
    session.logger.status("Barbed wire analysis job finished")
    if structure.deleted:
        raise UserError("AlphaFold structure was deleted during analysis")
    # Ininitially color them all dark gray, the "unassigned" color
    color_names = {
        "Predictive": "blue",
        "Unpacked high pLDDT": "gray",
        "Near-predictive": "green",
        "Unphysical": "purple",
        "Pseudostructure": "gold",
        "Barbed wire": "hotpink",
        "Unassigned": "dim gray",
    }
    from chimerax.core.colors import Color
    cat_colors = { cat: Color(color_name).uint8x4() for cat, color_name in color_names.items() }
    structure.residues.ribbon_colors = cat_colors["Unassigned"]

    from chimerax.atomic import Residue
    Residue.register_attr(session, "barbed_wire_category", "barbed wire", attr_type=str)
    for cat, res_infos in json['residues_by_category'].items():
        for chain, res_str in [res_info.split(',') for res_info in res_infos]:
            res_num = int(res_str.strip())
            res = structure.find_residue(chain, res_num)
            if res is None:
                session.logger.warning("Could not find residue %d of chain %s from barbed wire output in %s;"
                    " skipping" % (res_num, chain, structure))
                continue
            res.barbed_wire_category = cat
            try:
                res.ribbon_color = cat_colors[cat]
            except KeyError:
                raise RuntimeError("Unexpected structure category in barbed wire output: %s" % repr(cat))

    from chimerax.core.commands import run, StringArg
    run(session, "key %s pos 0.925,0.025 size 0.05,0.2 colorTreatment distinct labelSide left fontSize 16"
        % ' '.join([StringArg.unparse("%s:%s" % (color_names[cat], cat))
        for cat in reversed(sorted(list(color_names.keys())))]), log=False)

#NOTE: We don't use a REST server; reference code retained in douse.py

def _run_barbed_wire_subprocess(session, exe_path, optional_args, positional_args, temp_dir, verbose):
    '''
    Run barbed_wire_analysis in a subprocess and return the JSON output.
    '''
    from chimerax.core.commands import StringArg
    args = [exe_path, "model.pdb"] + ["output.type=json", "output.filename=barbed_wire_analysis_result.json"
        ] + optional_args + positional_args
    tsafe=session.ui.thread_safe
    logger = session.logger
    tsafe(logger.status, f'Running {exe_path} in directory {temp_dir}')
    import subprocess
    p = subprocess.run(args, capture_output = True, cwd = temp_dir)
    if p.returncode != 0:
        cmd = " ".join(args)
        out, err = p.stdout.decode("utf-8"), p.stderr.decode("utf-8")
        msg = (f'phenix.barbed_wire_analysis exited with error code {p.returncode}\n\n' +
               f'Command: {cmd}\n\n' +
               f'stdout:\n{out}\n\n' +
               f'stderr:\n{err}')
        raise UserError(msg)

    # Log command output
    if verbose:
        cmd = " ".join(args)
        out, err = p.stdout.decode("utf-8"), p.stderr.decode("utf-8")
        msg = f'<pre><b>Command</b>:\n\n{cmd}\n\n<b>stdout</b>:\n\n{out}'
        if err:
            msg += f'\n\n<b>stderr</b>:\n\n{err}'
        msg += '</pre>'
        tsafe(logger.info, msg, is_html=True)

    from os import path
    json_path = path.join(temp_dir, 'barbed_wire_analysis_result.json')
    if not path.exists(json_path):
        raise UserError('barbed_wire_anaylsis did not produce any JSON output')
    import json
    with open(json_path, 'r') as f:
        info = json.load(f)
    return info

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register
    from chimerax.core.commands import (EmptyArg, OpenFolderNameArg, BoolArg, RepeatOf, StringArg, Or)
    from chimerax.atomic import AtomicStructuresArg
    desc = CmdDesc(
        required = [('structures', Or(AtomicStructuresArg, EmptyArg)),],
        keyword = [
                   ('block', BoolArg),
                   ('key', BoolArg),
                   ('phenix_location', OpenFolderNameArg),
                   ('verbose', BoolArg),
                   ('option_arg', RepeatOf(StringArg)),
                   ('position_arg', RepeatOf(StringArg)),
        ],
        synopsis = 'Categorize regions of an AlphaFold prediction'
    )
    register('phenix barbedWire', desc, phenix_barbed_wire, logger=logger)
