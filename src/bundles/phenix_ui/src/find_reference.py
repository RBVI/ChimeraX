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
            positional_args, temp_dir, eff_save_dir, superimpose_phenix, verbose, callback, block):
        super().__init__(session)
        self._running = False
        self._monitor_time = 0
        self._monitor_interval = 10
        self.start(session, executable_location, optional_args, model_file_name, positional_args,
            eff_save_dir, superimpose_phenix, temp_dir, verbose, callback, blocking=block)

    def run(self, session, executable_location, optional_args, model_file_name, positional_args,
            eff_save_dir, superimpose_phenix, temp_dir, verbose, callback, **kw):
        self._running = True
        self.start_t = time()
        def threaded_run(self=self):
            try:
                reference = _run_find_ref_subprocess(session, executable_location, optional_args,
                    model_file_name, positional_args, eff_save_dir, superimpose_phenix, temp_dir, verbose)
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
def phenix_find_reference(session, chains, *, eff_save_directory=True, show_tool=True,
        superimpose_phenix=True, block=None, phenix_location=None, verbose=command_defaults['verbose'],
        option_arg=[], position_arg=[]):

    # Find the phenix.find_reference executable
    from .locate import find_phenix_command
    exe_path = find_phenix_command(session, 'phenix.find_reference', phenix_location)

    # if blocking not explicitly specified, block if in a script or in nogui mode
    if block is None:
        block = session.in_script or not session.ui.is_gui

    if not chains:
        raise UserError("No chains specified")

    import os
    if eff_save_directory is not False:
        possible_save_dirs = ['~/Desktop', '~'] if eff_save_directory is True else [eff_save_directory]
        for possible_save_dir in [os.path.expanduser(path) for path in possible_save_dirs]:
            if os.path.exists(possible_save_dir) and os.access(possible_save_dir, os.W_OK | os.X_OK):
                eff_save_directory = possible_save_dir
                break
        else:
            from chimerax.core.commands import plural_form, commas
            session.logger.warning("Cannot write to %s %s, not saving .eff file"
                % (plural_form(possible_save_dirs, "folder"), commas(possible_save_dirs)))
            eff_save_directory = False
    chains_by_structure = {}
    for structure, chain in zip(chains.structures, chains):
        chains_by_structure.setdefault(structure, []).append(chain)

    from chimerax.atomic import Chains
    for structure, chain_list in chains_by_structure.items():
        # Setup temporary directory to run phenix.find_reference
        from tempfile import TemporaryDirectory
        d = TemporaryDirectory(prefix = 'phenix_emis_')  # Will be cleaned up when object deleted.
        temp_dir = d.name

        # Save model to file.
        from chimerax.pdb import save_pdb
        from os import path
        prev_sel = structure.atoms.selecteds
        structure.atoms.selecteds = False
        Chains(chain_list).existing_residues.atoms.selecteds = True
        save_pdb(session, path.join(temp_dir,'model.pdb'), models=[structure], selected_only=True)
        structure.atoms.selecteds = prev_sel

        # Run phenix.find_reference
        # keep a reference to 'd' in the callback so that the temporary directory isn't removed before
        # the program runs
        callback = lambda json_info, *args, session=session, show_tool=show_tool, model=structure, \
            sup_phenix=superimpose_phenix, d_ref=d: _process_results(session, json_info, model, d.name,
            show_tool, sup_phenix)
        FindRefJob(session, exe_path, option_arg, "model.pdb", position_arg, temp_dir, eff_save_directory,
            superimpose_phenix, verbose, callback, block)

def _process_results(session, json_info, search_model, temp_dir, show_tool, superimpose_phenix):
    session.logger.status("Find-reference job finished")
    if search_model.deleted:
        raise UserError("Structure used as basis for search closed during search")

    from chimerax.core.models import Model
    ref_group = Model("%s reference structures" % search_model.name, session)
    session.models.add([ref_group])

    session.logger.info("Overall coverage: %g" % json_info['overall_coverage'])
    from chimerax.core.commands import run, StringArg
    from os import path
    collated_info = {}
    for result in json_info['results']:
        for k, v in result.items():
            if k == "file_name":
                target_cid = result['target_chain_id']
                ref_cid = result['reference_chain_id']
                identifier = "chain %s (%s)" % (target_cid, result['reference']['pdb_id'])
                for s in run(session, "open %s name %s inFileHistory false"
                        % (StringArg.unparse(path.join(temp_dir, v)), StringArg.unparse(identifier)),
                        log=False):
                    ref_group.add([s])
                    hide_spec = f"#{s.id_string} & ~ /{ref_cid}"
                    cmd = f"hide {hide_spec} ; ~cartoon {hide_spec}"
                    if not superimpose_phenix:
                        cmd += f"; matchmaker #{s.id_string}/{ref_cid} to" \
                            f" #{search_model.id_string}/{target_cid} logParameters false"
                    run(session, cmd, log=False)
                spec = '#%s/%s #%s/%s' % (search_model.id_string, target_cid, s.id_string, ref_cid)
                collated_info.setdefault('row name', []).append(
                    '<a href="cxcmd:view %s; sel %s">%s</a>' % (spec, spec, identifier))
            elif k in ('reference', 'calculated'):
                for subk, subv in v.items():
                    if subk == 'is_xray':
                        subk = 'is_experimental'
                    collated_info.setdefault(subk, []).append(subv)
            else:
                collated_info.setdefault(k, []).append(v)
    known_column_order = { 'main': [], 'reference': [], 'calculated': [] }
    known_columns = { k: set(v) for k,v in known_column_order.items() }
    # Don't want these as explicit table columns...
    known_columns['main'].update(('reference', 'calculated', 'file_name'))
    known_columns['reference'].update(('is_computational',))
    known_columns['calculated'].update(('xyz_pbs', 'tor_pbs', 'sort_value_1', 'sort_value_2'))
    column_order = { k: [] for k in known_column_order.keys()}
    for col_type, known_col_order in known_column_order.items():
        base = json_info['results'][0]
        col_dict = base if col_type == "main" else base[col_type]
        for kcol in known_col_order:
            if kcol in col_dict:
                column_order[col_type].append(kcol)
        known_cols = known_columns[col_type]
        for col_name in col_dict.keys():
            if col_name not in known_cols:
                if col_name == 'is_xray':
                    col_name = 'is_experimental'
                column_order[col_type].append(col_name)
    table_texts = []
    from chimerax.core.logger import html_table_params
    table_texts.append('<table %s>' % html_table_params)
    table_texts.append(' <thead>')
    table_texts.append('  <tr>')
    table_texts.append('   <th colspan="%d">Reference chain info</th>'
        % (len(column_order["main"]) + len(column_order["reference"]) + len(column_order["calculated"]) + 1))
    table_texts.append('  </tr>')
    table_texts.append(' </thead>')
    table_texts.append(' <tbody>')
    table_texts.append('  <tr>')
    table_texts.append('   ' +
        ' '.join(['<td style="text-align:center" rowspan="2">%s</td>' % item
            for item in (['model']+[col_name.replace('_', ' ') for col_name in column_order["main"]])]) + ' '
        + ' '.join(['<td style="text-align:center" colspan="%d">%s</td>' % (len(column_order[cat]), cat)
            for cat in ["reference", "calculated"]]))
    table_texts.append('  </tr>')
    table_texts.append('  <tr>')
    table_texts.append('   ' +
        ' '.join(['<td style="text-align:center">%s</td>' % item.replace('_', ' ')
            for item in column_order["reference"] + column_order["calculated"]]))
    table_texts.append('  </tr>')
    for i, row_name in enumerate(collated_info['row name']):
        table_texts.append('  <tr>')
        table_texts.append('   ' + ' '.join(['<td style="text-align:center">%s</td>' % process(item)
            for item in ([row_name]+[collated_info[col_name][i]
                for col_type in ('main', 'reference', 'calculated')
                for col_name in column_order[col_type]
                ])]))
        table_texts.append('  </tr>')
    table_texts.append(' </tbody>')
    table_texts.append('</table>')
    session.logger.info('\n'.join(table_texts), is_html=True)

#NOTE: We don't use a REST server; reference code retained in douse.py

def process(item):
    if isinstance(item, float) and len(str(item)) > 7 and str(item)[-5:].isdigit():
        return "%.4f" % item
    return item

def _run_find_ref_subprocess(session, exe_path, optional_args, model_file_name, positional_args,
        eff_save_dir, superimpose_phenix, temp_dir, verbose):
    '''
    Run find_reference in a subprocess and return the reference information.
    '''
    super_arg = [] if superimpose_phenix else ["superpose_reference_on_target=False"]
    from chimerax.core.commands import StringArg
    args = [exe_path] + optional_args + [
            "--json-filename", "find_reference.json",
            StringArg.unparse(model_file_name),
        ] + super_arg + positional_args
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

    from os import path, listdir, rename

    # Save EFF file
    if eff_save_dir is not False:
        for fname in listdir(temp_dir):
            if fname.endswith(".eff"):
                try:
                    rename(path.join(temp_dir, fname), path.join(eff_save_dir, fname))
                except OSError as e:
                    logger.warning("Could not save .eff file to %s; the error message was: %s"
                        % (eff_save_dir, str(e)))
                else:
                    logger.info("Saved EFF file to %s" % path.join(eff_save_dir, fname))

    # Return JSON information
    json_path = path.join(temp_dir,'find_reference.json')
    import json
    with open(json_path, 'r') as f:
        info = json.load(f)
    return info

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register
    from chimerax.core.commands import (CenterArg, OpenFolderNameArg, BoolArg, NonNegativeFloatArg,
        Or, RepeatOf, StringArg, SaveFolderNameArg)
    from chimerax.map import MapArg, MapsArg
    from chimerax.atomic import UniqueChainsArg, AtomicStructuresArg
    desc = CmdDesc(
        required = [('chains', UniqueChainsArg),
        ],
        keyword = [('block', BoolArg),
                   ('eff_save_directory', Or(BoolArg, SaveFolderNameArg)),
                   ('phenix_location', OpenFolderNameArg),
                   ('superimpose_phenix', BoolArg),
                   ('verbose', BoolArg),
                   ('option_arg', RepeatOf(StringArg)),
                   ('position_arg', RepeatOf(StringArg)),
                   ('show_tool', BoolArg),
        ],
        synopsis = 'Find reference structure'
    )
    register('phenix findReference', desc, phenix_find_reference, logger=logger)
