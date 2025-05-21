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
# Command to fit loops guided by cryoEM maps using Phenix fit_loops.
#
from chimerax.core.tasks import Job
from chimerax.core.errors import UserError, LimitationError
from time import time

class NoSeqInfoError(UserError):
    pass

class FitLoopsJob(Job):

    SESSION_SAVE = False

    def __init__(self, session, executable_location, optional_args, map_file_name, model_file_name,
            sequence_file_name, positional_args, temp_dir, start_res_number, end_res_number, chain_id,
            processors, verbose, callback, block):
        super().__init__(session)
        self._running = False
        self._monitor_time = 0
        self._monitor_interval = 10
        self.start(session, executable_location, optional_args, map_file_name, model_file_name,
            sequence_file_name, positional_args, temp_dir, start_res_number, end_res_number, chain_id,
            processors, verbose, callback, blocking=block)

    def run(self, session, executable_location, optional_args, map_file_name, model_file_name,
            sequence_file_name, positional_args, temp_dir, start_res_number, end_res_number, chain_id,
            processors, verbose, callback, **kw):
        self._running = True
        self.start_t = time()
        def threaded_run(self=self):
            try:
                results = _run_fit_loops_subprocess(session, executable_location, optional_args,
                    map_file_name, model_file_name, sequence_file_name, positional_args, temp_dir,
                    start_res_number, end_res_number, chain_id, processors, verbose)
            except Exception as e:
                from .util import thread_throw
                thread_throw(session, e)
                return
            finally:
                self._running = False
            if results:
                self.session.ui.thread_safe(callback, *results)
        import threading
        thread = threading.Thread(target=threaded_run, daemon=True)
        thread.start()
        super().run()

    def monitor(self):
        delta = int(time() - self.start_t + 0.5)
        from chimerax.core.commands import plural_form
        if delta < 60:
            time_info = "%d %s" % (delta, plural_form(delta, "second"))
        elif delta < 3600:
            minutes = delta // 60
            seconds = delta % 60
            time_info = "%d %s and %d %s" % (minutes, plural_form(minutes, "minute"), seconds,
                plural_form(seconds, "second"))
        else:
            hours = delta // 3600
            minutes = (delta % 3600) // 60
            seconds = delta % 60
            time_info = "%d:%02d:%02d" % (hours, minutes, seconds)
        ses = self.session
        ses.ui.thread_safe(ses.logger.status, "Fit loops job still running (%s)" % time_info)

    def next_check(self):
        return self._monitor_interval
        self._monitor_time += self._monitor_interval
        return self._monitor_time

    def running(self):
        return self._running

command_defaults = {
    'processors': None,
    'replace': True,
    'verbose': False
}
def phenix_fit_loops(session, residues, in_map, *, block=None, gap_only=False, phenix_location=None,
        processors=command_defaults['processors'], replace=command_defaults['replace'],
        sequence_file=None, verbose=command_defaults['verbose'], option_arg=[], position_arg=[]):

    from chimerax.atomic import Residue
    residues = residues.filter(residues.polymer_types == Residue.PT_AMINO)
    if not residues:
        raise UserError(
            "No protein residues specfied. Make sure your atom specifier includes *existing* residues,\n"
            "so if filling in missing structure, you need to specify the residues adjacent to the gap.")

    # Find the phenix.fit_loops executable
    from .locate import find_phenix_command
    exe_path = find_phenix_command(session, 'phenix.fit_loops', phenix_location)

    # if blocking not explicitly specified, block if in a script or in nogui mode
    if block is None:
        block = session.in_script or not session.ui.is_gui

    if processors is None:
        import os
        if hasattr(os, 'sched_getaffinity'):
            processors = max(1, len(os.sched_getaffinity(0)))
        else:
            processors = os.cpu_count()
            if processors is None:
                processors = 1
    from chimerax.core.commands import plural_form
    session.logger.info("Using %d %s" % (processors, plural_form(processors, "processor")))

    def seq_num(r):
        if r.insertion_code:
            raise LimitationError("phenix.fit_loops cannot handles residues with insertion codes"
                " (i.e. %s)" % r)
        return r.number
    # process the residues into phenix-fit-friendly units
    job_info = []
    for structure, s_residues in residues.by_structure:
        if gap_only:
            try:
                pbs = structure.pbg_map[structure.PBG_MISSING_STRUCTURE].pseudobonds
            except KeyError:
                session.logger.warning("Structure %s has no missing residues!" % structure)
                continue
            res_set = set(s_residues)
            for pb in pbs:
                a1, a2 = pb.atoms
                r1, r2 = (a1.residue, a2.residue) if a1 < a2 else (a2.residue, a1.residue)
                if r1 == r2:
                    continue
                if r1 in res_set and r2 in res_set:
                    job_info.append((seq_num(r1)+1, seq_num(r2)-1, r1.chain_id, False))
        else:
            # remodelling at least some existing residues; figure out runs of residues in the same chain
            for chain_id in residues.unique_chain_ids:
                res_list = residues.filter(residues.chain_ids == chain_id)
                req_chain_residues = set(res_list)
                start = end = None
                for r in res_list[0].chain.existing_residues:
                    if r in req_chain_residues:
                        if start is None:
                            start = end = r
                        else:
                            end = r
                    else:
                        if start is not None:
                            job_info.append((seq_num(start), seq_num(end), chain_id, True))
                            start = end = None
                if start is not None:
                    job_info.append((seq_num(start), seq_num(end), chain_id, True))

        session.logger.info("Running %d %s" % (len(job_info), plural_form(job_info, "job")))

        if block:
            procs_per_job = [processors] * len(job_info)
        else:
            per_job = int(processors / len(job_info))
            if per_job <= 1:
                procs_per_job = [1] * len(job_info)
            elif len(job_info) == 1:
                procs_per_job = [processors]
            else:
                procs_per_job = [per_job] * (len(job_info)-1)
                procs_per_job.append(processors - sum(procs_per_job))

        for start_res_number, end_res_number, chain_id, remove_loops in job_info:
            # Setup temporary directory to run phenix.fit_loops.
            from tempfile import TemporaryDirectory
            d = TemporaryDirectory(prefix = 'phenix_fit_loops_')  # Will be cleaned up when object deleted.
            temp_dir = d.name

            # Save map to file
            from os import path
            from chimerax.map_data import save_grid_data
            save_grid_data([in_map.data], path.join(temp_dir,'map.mrc'), session)

            # Save structure to file.
            from chimerax.pdb import save_pdb
            save_pdb(session, path.join(temp_dir,'model.pdb'), models=[structure], rel_model=map)

            seqf_path = path.join(temp_dir, "sequences")
            if sequence_file is None:
                with open(seqf_path, "w") as f:
                    for chain in structure.chains:
                        if not chain.full_sequence_known:
                            raise NoSeqInfoError("Structure file does not contain complete sequence"
                                f" information.  Please provide that information via the '{seq_keyword}'"
                                " keyword argument.")
                        print("> Chain %s" % chain.chain_id, file=f)
                        print(chain.characters, file=f)
            else:
                import shutil
                shutil.copyfile(sequence_file, seqf_path)

            # Run phenix.fit_loops
            # keep a reference to 'd' in the callback so that the temporary directory isn't removed before
            # fit_loops runs
            callback = lambda fit_loops_model, info, *args, session=session, shift=0, \
                structure=structure, map=in_map, start_res_number=start_res_number, \
                end_res_number=end_res_number, replace=replace, chain_id=chain_id, d_ref=d: \
                _process_results(session, fit_loops_model, map, shift, structure, start_res_number, \
                end_res_number, replace, chain_id, info)
            FitLoopsJob(session, exe_path, option_arg + (["remove_loops=True"] if remove_loops else []),
                "map.mrc", "model.pdb", "sequences", position_arg, temp_dir, start_res_number,
                end_res_number, chain_id, procs_per_job.pop(), verbose, callback, block)

def get_missing_structure_labels_model(structure):
    pbg = structure.pseudobond_group(structure.PBG_MISSING_STRUCTURE)
    if not pbg:
        return None
    from chimerax.label.label3d import ObjectLabels
    for cm in pbg.child_models():
        if isinstance(cm, ObjectLabels):
            return cm
    return None

def get_missing_structure_labels(structure):
    labels_model = get_missing_structure_labels_model(structure)
    if not labels_model:
        return set()
    return set(labels_model.labels())

def _process_results(session, fit_loops_model, map, shift, structure, start_res_number, end_res_number,
        replace, chain_id, info):
    fit_loops_model.position = map.scene_position
    if shift is not None:
        fit_loops_model.atoms.coords += shift

    if replace:
        undo_all_atoms = structure.atoms
        undo_all_coords = undo_all_atoms.coords
        # Only "best effort" to restore pseudobond label on undo: label group must already exist at undo
        # time and only one pseudobond gets created by the undo
        # 
        # Get label attrs before pseudobond gets deleted, since default values may ask for attributes
        # of the pseudobond
        labels = get_missing_structure_labels(structure)
        label_infos = {
            label: { name: getattr(label, name) for name in ['view', 'offset', 'text', 'color',
                'background', 'attribute', 'size', 'height', 'font'] } for label in labels
        }

        orig_atom_map = dict([(a.string(style="simple", omit_structure=True), a) for a in structure.atoms])
        orig_res_map = dict([(r.string(style="simple", omit_structure=True), r) for r in structure.residues])
        fit_res_indices = dict([(r, i) for i, r in enumerate(fit_loops_model.residues)])
        chain_sequence = dict([(c.chain_id, c.characters) for c in structure.chains])
        new_residues = []
        new_atoms = []
        from chimerax.atomic.struct_edit import add_atom, add_bond
        for fit_atom in fit_loops_model.atoms:
            key = fit_atom.string(style="simple", omit_structure=True)
            try:
                orig_atom = orig_atom_map[key]
            except KeyError:
                fit_res = fit_atom.residue
                r_key = fit_res.string(style="simple", omit_structure=True)
                try:
                    orig_res = orig_res_map[r_key]
                except KeyError:
                    for follower in fit_loops_model.residues[fit_res_indices[fit_res]+1:]:
                        try:
                            precedes = orig_res_map[follower.string(style="simple", omit_structure=True)]
                        except KeyError:
                            continue
                        break
                    else:
                        precedes = None
                    orig_res = orig_res_map[r_key] = structure.new_residue(fit_res.name, fit_res.chain_id,
                        fit_res.number, insert=fit_res.insertion_code, precedes=precedes)
                    new_residues.append(orig_res)
                orig_atom_map[key] = add_atom(fit_atom.name, fit_atom.element, orig_res, fit_atom.coord,
                    bfactor=fit_atom.bfactor)
                new_atoms.append((fit_atom, orig_atom_map[key]))
            else:
                orig_atom.coord = fit_atom.coord
        # add bonds
        bonded_residues = set()
        gap_residues = set()
        for fit_atom, orig_atom in new_atoms:
            gap_residues.add(orig_atom.residue)
            for fnb in fit_atom.neighbors:
                onb = orig_atom_map[fnb.string(style="simple", omit_structure=True)]
                bonded_residues.add(onb.residue)
                if onb not in orig_atom.neighbors:
                    add_bond(orig_atom, onb)
        # check that sequences did not change
        for chain in structure.chains:
            cur_seq = chain.characters
            prev_seq = chain_sequence[chain.chain_id]
            if cur_seq != prev_seq:
                for i1 in range(len(cur_seq)):
                    if cur_seq[i1] != prev_seq[i1]:
                        break
                for i2 in range(len(cur_seq)):
                    if cur_seq[-(i2+1)] != prev_seq[-(i2+1)]:
                        break
                session.logger.info("Sequence of chain %s before fit_loops (differences capitalized):"
                    % chain.chain_id)
                session.logger.info("%s%s%s"
                    % (prev_seq[:i1].lower(), prev_seq[i1:-i2], prev_seq[-i2:].lower()))
                session.logger.info("Sequence of chain %s after fit_loops (differences capitalized):"
                    % chain.chain_id)
                session.logger.info("%s%s%s"
                    % (cur_seq[:i1].lower(), cur_seq[i1:-i2], cur_seq[-i2:].lower()))
                for fl_chain in fit_loops_model.chains:
                    if fl_chain.chain_id == chain.chain_id:
                        session.logger.info("Raw fit_loops chain %s sequence:" % chain.chain_id)
                        session.logger.info(fl_chain.characters.lower())
                        break
                for r in new_residues:
                    structure.delete_residue(r)
                raise AssertionError("Chain sequence for chain %s changed after fit_loops; undoing changes"
                    % chain.chain_id)

        dead_labels = [lbl for lbl in labels if lbl.object.deleted]
        if len(dead_labels) == 1:
            label = dead_labels.pop()
            label_info = label_infos[label]
        else:
            label_info = None

        # show residues adjacent to the gap as stick
        from chimerax.atomic import Residues, Atoms, Atom
        edge_residues = Residues(bonded_residues - gap_residues)
        edge_ribbon_displays = edge_residues.ribbon_displays
        edge_residues.ribbon_displays = False
        edge_atoms = edge_residues.atoms
        edge_atom_displays = edge_atoms.displays
        edge_atom_draw_modes = edge_atoms.draw_modes
        edge_atoms.displays = True
        edge_atoms.draw_modes = Atom.STICK_STYLE
        from chimerax.core.undo import UndoAction
        class FitLoopsUndo(UndoAction):
            def __init__(self, session, data=(undo_all_atoms, undo_all_coords, edge_residues,
                    edge_ribbon_displays, edge_atoms, edge_atom_displays, edge_atom_draw_modes,
                    Atoms([orig for fit, orig in new_atoms]), label_info)):
                self.session = session
                self.data = data
                super().__init__("Fit Loops result", can_redo=False)

            def undo(self):
                all_atoms, all_coords, edge_residues, edge_ribbon_displays, edge_atoms, \
                    edge_atom_displays, edge_atom_draw_modes, new_atoms, label_info = self.data
                if not all_atoms:
                    return
                if label_info:
                    structure = all_atoms[0].structure
                    labels_model = get_missing_structure_labels_model(structure)
                    if labels_model:
                        ms_pbg = structure.pseudobond_group(structure.PBG_MISSING_STRUCTURE)
                        if ms_pbg:
                            orig_pbs = set(ms_pbg.pseudobonds)
                        else:
                            orig_pbs = set()
                if new_atoms:
                    for a in new_atoms:
                        a.structure.delete_atom(a)
                if len(all_atoms) == len(all_coords):
                    all_atoms.coords = all_coords
                else:
                    self.session.logger.warning("Not all previous atoms still exist; not restoring"
                        " previous coordinates")
                if len(edge_residues) == len(edge_ribbon_displays):
                    edge_residues.ribbon_displays = edge_ribbon_displays
                    if len(edge_atoms) == len(edge_atom_displays):
                        edge_atoms.displays = edge_atom_displays
                        edge_atoms.draw_modes = edge_atom_draw_modes
                    else:
                        self.session.logger("Not all atoms in residues adjacent to gap still exist;"
                            " not restoring their display status and draw mode")
                else:
                    self.session.logger.warning("Not all residues adjacent to gap still exist; not"
                        " restoring their state")
                if label_info and labels_model:
                    if ms_pbg:
                        cur_pbs = set(ms_pbg.pseudobonds)
                    else:
                        cur_pbs = set()
                    diff_pbs = cur_pbs - orig_pbs
                    if len(diff_pbs) == 1:
                        pb = diff_pbs.pop()
                        view, offset, text, color, background, attribute, size, height, font = label_info
                        from chimerax.label.label3d import PseudobondLabel
                        labels_model.add_labels([pb], PseudobondLabel, label_info.pop('view'),
                            settings=label_info)

        session.undo.register(FitLoopsUndo(session))
        fit_loops_model.delete()
    else:
        fit_loops_model.position = structure.scene_position
        session.models.add([fit_loops_model])
    if session.ui.is_gui:
        from .tool import FitLoopsResultsViewer
        FitLoopsResultsViewer(session, structure if replace else fit_loops_model, info, map)
    else:
        print("Fit loops JSON output:", info)

def _run_fit_loops_subprocess(session, exe_path, optional_args, map_filename, model_filename, seq_filename,
        positional_args, temp_dir, start_res_number, end_res_number, chain_id, processors, verbose):
    '''
    Run fit_loops in a subprocess and return the model with predicted waters.
    '''
    output_file = "fl_out.pdb"
    json_file = "fit_loops.json"
    args = [exe_path] + optional_args + [f"map_in={map_filename}", f"pdb_in={model_filename}",
        f"seq_file={seq_filename}", f"nproc={processors}", f"pdb_out={output_file}",
        "results_as_json=" + json_file, "loop_backup_residues=0"] + positional_args
    if start_res_number is not None:
        args += [f"start={start_res_number}"]
    if end_res_number is not None:
        args += [f"end={end_res_number}"]
    if chain_id is not None:
        args += [f"chain_id={chain_id}"]
    tsafe=session.ui.thread_safe
    logger = session.logger
    tsafe(logger.status, f'Running {exe_path} in directory {temp_dir}')
    import subprocess
    p = subprocess.run(args, capture_output = True, cwd = temp_dir)
    def raise_in_main_thread(msg):
        from chimerax.core.errors import NonChimeraXError
        raise NonChimeraXError(msg)
    if p.returncode != 0:
        cmd = " ".join(args)
        out, err = p.stdout.decode("utf-8"), p.stderr.decode("utf-8")
        msg = (f'phenix.fit_loops exited with error code {p.returncode}\n\n' +
               f'Command: {cmd}\n\n' +
               f'stdout:\n{out}\n\n' +
               f'stderr:\n{err}')
        msg = f'phenix.fit_loops exited with error code {p.returncode}<br><br>'
        msg += f'<pre><b>Command</b>:\n\n{cmd}\n\n<b>stdout</b>:\n\n{out}'
        if err:
            msg += f'\n\n<b>stderr</b>:\n\n{err}'
        msg += '</pre>'
        tsafe(logger.info, msg, is_html=True)
        tsafe(raise_in_main_thread, "phenix.fit_loops failed; see log for details")
        return None

    # Log phenix fit_loops command output
    if verbose:
        cmd = " ".join(args)
        out, err = p.stdout.decode("utf-8"), p.stderr.decode("utf-8")
        msg = f'<pre><b>Command</b>:\n\n{cmd}\n\n<b>stdout</b>:\n\n{out}'
        if err:
            msg += f'\n\n<b>stderr</b>:\n\n{err}'
        msg += '</pre>'
        tsafe(logger.info, msg, is_html=True)

    # gather JSON info
    from os import path, listdir
    json_path = path.join(temp_dir, json_file)
    import json
    with open(json_path, 'r') as f:
        info = json.load(f)

    # Open new model with added loops
    output_path = path.join(temp_dir, output_file)
    if not path.exists(output_path):
        cmd = " ".join(args)
        out, err = p.stdout.decode("utf-8"), p.stderr.decode("utf-8")
        msg = f'<pre><b>Command</b>:\n\n{cmd}\n\n<b>stdout</b>:\n\n{out}'
        if err:
            msg += f'\n\n<b>stderr</b>:\n\n{err}'
        msg += '</pre>'
        tsafe(logger.info, msg, is_html=True)
        tsafe(raise_in_main_thread, "fit_loops did not find viable loop(s); see log for details")
        return None
    from chimerax.core.logger import PlainTextLog
    class IgnoreBlankLinesLog(PlainTextLog):
        excludes_other_logs = True
        def log(self, level, msg):
            return msg.startswith("Ignored bad PDB record")
    log = IgnoreBlankLinesLog()
    session.logger.add_log(log)
    from chimerax.pdb import open_pdb
    try:
        models, status_info = open_pdb(session, output_path, log_info = False)
    finally:
        session.logger.remove_log(log)

    return models[0], info

seq_keyword = 'sequence_file'
def register_command(logger):
    from chimerax.core.commands import CmdDesc, register
    from chimerax.core.commands import OpenFolderNameArg, OpenFileNameArg, BoolArg, FloatArg, RepeatOf
    from chimerax.core.commands import StringArg, PositiveIntArg
    from chimerax.map import MapArg
    from chimerax.atomic import ResiduesArg
    desc = CmdDesc(
        required = [('residues', ResiduesArg)],
        keyword = [('in_map', MapArg),
                   ('block', BoolArg),
                   ('processors', PositiveIntArg),
                   ('replace', BoolArg),
                   (seq_keyword, OpenFileNameArg),
                   ('verbose', BoolArg),
                   ('phenix_location', OpenFolderNameArg),
                   ('option_arg', RepeatOf(StringArg)),
                   ('position_arg', RepeatOf(StringArg)),
                   ('gap_only', BoolArg),
        ],
        required_arguments = ['in_map'],
        synopsis = 'Fit loop(s) into density'
    )
    register('phenix fitLoops', desc, phenix_fit_loops, logger=logger)
