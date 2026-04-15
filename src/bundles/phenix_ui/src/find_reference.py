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
# Command to place a structure in a cryoEM map using Phenix emplace_local.
#
from chimerax.core.tasks import Job
from chimerax.core.errors import UserError
from time import time

class FindRefJob(Job):

    SESSION_SAVE = False

    def __init__(self, session, executable_location, optional_args, model_file_name,
            positional_args, temp_dir, verbose, callback, block):
        super().__init__(session)
        self._running = False
        self._monitor_time = 0
        self._monitor_interval = 10
        self.start(session, executable_location, optional_args, model_file_name, positional_args,
            temp_dir, verbose, callback, blocking=block)

    def run(self, session, executable_location, optional_args, model_file_name, positional_args,
            temp_dir, verbose, callback, **kw):
        self._running = True
        self.start_t = time()
        def threaded_run(self=self):
            try:
                reference = _run_find_ref_subprocess(session, executable_location, optional_args,
                    model_file_name, positional_args, temp_dir, verbose)
            except Exception as e:
                from .util import thread_throw
                thread_throw(session, e)
                return
            finally:
                self._running = False
            self.session.ui.thread_safe(callback, reference)
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
            if seconds != 0:
                time_info = "%d %s and %d %s" % (minutes, plural_minutes(minutes), seconds,
                    plural_seconds(seconds))
            else:
                time_info = "%d %s" % (minutes, plural_minutes(minutes))
        else:
            hours = delta // 3600
            minutes = (delta % 3600) // 60
            seconds = delta % 60
            time_info = "%d:%02d:%02d" % (hours, minutes, seconds)
        ses = self.session
        ses.ui.thread_safe(ses.logger.status, "Find reference job still running (%s)" % time_info)

    def next_check(self):
        return self._monitor_interval
        self._monitor_time += self._monitor_interval
        return self._monitor_time

    def running(self):
        return self._running

command_defaults = {
    'verbose': False
}
def phenix_find_reference(session, model, *, show_tool=True, block=None, phenix_location=None,
        verbose=command_defaults['verbose'], option_arg=[], position_arg=[]):

    # Find the phenix.find_reference executable
    from .locate import find_phenix_command
    exe_path = find_phenix_command(session, 'phenix.find_reference', phenix_location)

    # if blocking not explicitly specified, block if in a script or in nogui mode
    if block is None:
        block = session.in_script or not session.ui.is_gui

    # Setup temporary directory to run phenix.find_reference
    from tempfile import TemporaryDirectory
    d = TemporaryDirectory(prefix = 'phenix_emis_')  # Will be cleaned up when object deleted.
    temp_dir = d.name

    # Save model to file.
    from chimerax.pdb import save_pdb
    from os import path
    save_pdb(session, path.join(temp_dir,'model.pdb'), models=[model])

    # Run phenix.find_reference
    # keep a reference to 'd' in the callback so that the temporary directory isn't removed before
    # the program runs
    callback = lambda json_info, *args, session=session, show_tool=show_tool, model=model, d_ref=d: \
        _process_results(session, json_info, model, d.name, show_tool)
    FindRefJob(session, exe_path, option_arg, "model.pdb", position_arg, temp_dir, verbose, callback, block)

def _process_results(session, json_info, search_model, temp_dir, show_tool):
    session.logger.status("Find-reference job finished")
    if search_model.deleted:
        raise UserError("Structure used as basis for search closed during search")

    print(json_info)

    from chimerax.core.models import Model
    ref_group = Model("%s reference structures" % search_model.name, session)
    session.models.add([ref_group])

    session.logger.info("Overall coverage: %g" % json_info['overall_coverage'])
    from chimerax.core.commands import run, StringArg
    from os import path
    collated_info = {}
    known_column_order = []
    known_columns = set(known_column_order)
    column_order = []
    for known_col in known_column_order:
        if known_col in json_info['results'][0]:
            column_order.append(known_col)
    for col in json_info['results'][0].keys():
        if col not in known_columns and col != 'file_name':
            column_order.append(col)
    for result in json_info['results']:
        for k, v in result.items():
            if k == "file_name":
                target_cid = result['target_chain_id']
                ref_cid = result['reference_chain_id']
                identifier = "chain %s (%s)" % (target_cid, result['reference']['pdb_id'])
                for s in run(session, "open %s name %s" % (StringArg.unparse(path.join(temp_dir, v)),
                        StringArg.unparse(identifier)), log=False):
                    ref_group.add([s])
                    hide_spec = f"#{s.id_string} & ~ /{ref_cid}"
                    run(session, f"hide {hide_spec} ; ~cartoon {hide_spec} ;"
                        f" matchmaker #{s.id_string}/{ref_cid} to #{search_model.id_string}/{target_cid}")
                collated_info.setdefault('row name', []).append(identifier)
            else:
                collated_info.setdefault(k, []).append(v)
    table_texts = []
    from chimerax.core.logger import html_table_params
    table_texts.append('<table %s>' % html_table_params)
    table_texts.append(' <thead>')
    table_texts.append('  <tr>')
    table_texts.append('   <th colspan="%d">Reference chain info</th>' % (len(column_order)+1))
    table_texts.append('  </tr>')
    table_texts.append(' </thead>')
    table_texts.append(' <tbody>')
    table_texts.append('  <tr>')
    table_texts.append('   ' + ' '.join(['<td style="text-align:center">%s</td>' % item
        for item in (['model']+[col_name.replace('_', ' ') for col_name in column_order])]))
    table_texts.append('  </tr>')
    for i, row_name in enumerate(collated_info['row name']):
        table_texts.append('  <tr>')
        table_texts.append('   ' + ' '.join(['<td style="text-align:center">%s</td>' % item
            for item in ([row_name]+[collated_info[col_name][i] for col_name in column_order])]))
        table_texts.append('  </tr>')
    table_texts.append(' </tbody>')
    table_texts.append('</table>')
    session.logger.info('\n'.join(table_texts), is_html=True)

#NOTE: We don't use a REST server; reference code retained in douse.py

def _run_find_ref_subprocess(session, exe_path, optional_args, model_file_name, positional_args,
        temp_dir, verbose):
    '''
    Run find_reference in a subprocess and return the model.
    '''
    from chimerax.core.commands import StringArg
    args = [exe_path] + optional_args + [
            "--json-filename", "find_reference.json",
            StringArg.unparse(model_file_name),
        ] + positional_args
    tsafe=session.ui.thread_safe
    logger = session.logger
    tsafe(logger.status, f'Running {exe_path} in directory {temp_dir}')
    import subprocess
    p = subprocess.run(args, capture_output = True, cwd = temp_dir)
    if p.returncode != 0:
        cmd = " ".join(args)
        out, err = p.stdout.decode("utf-8"), p.stderr.decode("utf-8")
        msg = (f'phenix.find_reference exited with error code {p.returncode}\n\n' +
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

    # Open new model with added waters
    from os import path
    json_path = path.join(temp_dir,'find_reference.json')
    import json
    with open(json_path, 'r') as f:
        info = json.load(f)
    return info

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register
    from chimerax.core.commands import (CenterArg, OpenFolderNameArg, BoolArg, NonNegativeFloatArg,
        RepeatOf, StringArg)
    from chimerax.map import MapArg, MapsArg
    from chimerax.atomic import AtomicStructureArg, AtomicStructuresArg
    desc = CmdDesc(
        required = [('model', AtomicStructureArg),
        ],
        keyword = [('block', BoolArg),
                   ('phenix_location', OpenFolderNameArg),
                   ('verbose', BoolArg),
                   ('option_arg', RepeatOf(StringArg)),
                   ('position_arg', RepeatOf(StringArg)),
                   ('show_tool', BoolArg),
        ],
        synopsis = 'Find reference structure'
    )
    register('phenix findReference', desc, phenix_find_reference, logger=logger)
