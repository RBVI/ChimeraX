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

# ==============================================================================
# Code for formatting listinfo responses

def attr_string(obj, attr):
    # Get attribute as a string
    a = getattr(obj, attr)
    if isinstance(a, str):
        # String, no action needed
        s = a
    elif isinstance(a, bytes) or isinstance(a, bytearray):
        # Binary, decode into string
        s = a.decode("utf-8")
    else:
        try:
            # Sequence, make into comma-separated string
            s = ','.join(str(v) for v in a)
        except TypeError:
            # Something else, convert to string
            s = str(a)
    # Convert into double-quote string if necessary
    l = ['"']
    need_quotes = False
    for c in s:
        if c in '"\\':
            l.append('\\')
        elif c in ' \t':
            need_quotes = True
        l.append(c)
    if need_quotes:
        l.append('"')
        return ''.join(l)
    else:
        return s

def spec(o):
    try:
        return o.atomspec
    except AttributeError:
        try:
            return '#' + o.id_string
        except AttributeError:
            return ""

def report_models(logger, models, attr, *, return_json=False, save_file=None):
    msgs = []
    for m in models:
        try:
            value = attr_string(m, attr)
        except AttributeError:
            value = "[undefined]"
        msgs.append("model id %s type %s %s %s" % (spec(m), type(m).__name__,
                                                   attr, value))
    output(logger, save_file, "\n".join(msgs))
    if return_json:
        model_infos = []
        for model in models:
            present = True
            try:
                val = getattr(model, attr)
            except AttributeError:
                present = False
                val = None
            model_infos.append({
                'spec': model.atomspec,
                'class': model.__class__.__name__,
                'attribute': attr,
                'present': present,
                'value': val
            })
        from chimerax.core.commands import JSONResult, ArrayJSONEncoder
        import json
        return JSONResult(ArrayJSONEncoder().encode(model_infos), None)

def report_chains(logger, chains, attr, *, return_json=False, save_file=None):
    msgs = []
    for c in chains:
        try:
            value = attr_string(c, attr)
        except AttributeError:
            continue
        msgs.append("chain id %s %s %s" % (spec(c), attr, value))
    output(logger, save_file, "\n".join(msgs))
    if return_json:
        from chimerax.atomic import Residue
        chain_infos = []
        for chain in chains:
            present = True
            try:
                val = getattr(chain, attr)
            except AttributeError:
                present = False
                val = None
            chain_infos.append({
                'spec': chain.atomspec,
                'attribute': attr,
                'sequence': chain.characters,
                'residues': [r.atomspec if r else None for r in chain.residues],
                'polymer type': 'nucleic' if chain.polymer_type == Residue.PT_NUCLEIC else "protein",
                'present': present,
                'value': val
            })
        from chimerax.core.commands import JSONResult, ArrayJSONEncoder
        import json
        return JSONResult(ArrayJSONEncoder().encode(chain_infos), None)

def report_polymers(logger, polymers, *, return_json=False, save_file=None):
    msgs = []
    for p in polymers:
        if len(p) < 2:
            continue
        msgs.append("physical chain %s %s" % (spec(p[0]), spec(p[-1])))
    output(logger, save_file, "\n".join(msgs))
    if return_json:
        polymer_infos = []
        for polymer in polymers:
            if len(polymer) < 2:
                continue
            polymer_infos.append([r.atomspec for r in polymer])
        from chimerax.core.commands import JSONResult, ArrayJSONEncoder
        import json
        return JSONResult(ArrayJSONEncoder().encode(polymer_infos), None)

def report_residues(logger, residues, attr, *, return_json=False, save_file=None):
    msgs = []
    for r in residues:
        try:
            value = attr_string(r, attr)
        except AttributeError:
            continue
        info = "residue id %s %s %s" % (spec(r), attr, value)
        try:
            index = r.chain.residues.index(r)
        except (AttributeError, ValueError):
            pass
        else:
            info += " index %s" % index
        msgs.append(info)
    output(logger, save_file, "\n".join(msgs))
    if return_json:
        residue_infos = []
        for r in residues:
            present = True
            try:
                val = getattr(r, attr)
            except AttributeError:
                present = False
                val = None
            residue_infos.append({
                'spec': r.atomspec,
                'attribute': attr,
                'present': present,
                'value': val
            })
        from chimerax.core.commands import JSONResult, ArrayJSONEncoder
        import json
        return JSONResult(ArrayJSONEncoder().encode(residue_infos), None)

def report_atoms(logger, atoms, attr, *, return_json=False, save_file=None):
    msgs = []
    for a in atoms:
        try:
            value = attr_string(a, attr)
        except AttributeError:
            pass
        else:
            msgs.append("atom id %s %s %s" % (spec(a), attr, value))
    output(logger, save_file, "\n".join(msgs))
    if return_json:
        atom_infos = []
        for a in atoms:
            present = True
            try:
                val = getattr(a, attr)
            except AttributeError:
                present = False
                val = None
            atom_infos.append({
                'spec': a.atomspec,
                'attribute': attr,
                'present': present,
                'value': val
            })
        from chimerax.core.commands import JSONResult, ArrayJSONEncoder
        import json
        return JSONResult(ArrayJSONEncoder().encode(atom_infos), None)

def report_attr(logger, prefix, attr, *, save_file=None, append=False):
    output(logger, save_file, "%sattr %s" % (prefix, attr), append=append)

def report_distmat(logger, atoms, distmat, *, save_file=None):
    num_atoms = len(atoms)
    msgs = []
    for i in range(num_atoms):
        for j in range(i+1,num_atoms):
            # distmat is a scipy condensed distance matrix
            # Index calculation from answer by HongboZhu in
            # https://stackoverflow.com/questions/13079563/how-does-condensed-distance-matrix-work-pdist
            dmi = num_atoms*i - i*(i+1)//2 + j - 1 - i
            msgs.append("distmat %s %s %s" % (spec(atoms[i]), spec(atoms[j]),
                                              distmat[dmi]))
    output(logger, save_file, '\n'.join(msgs))

def output(logger, file_name, msg, *, append=False):
    if file_name:
        from chimerax.io import open_output
        with open_output(file_name, 'utf-8', append=append) as outf:
            outf.write(msg + ('' if msg.endswith('\n') else '\n'))
    else:
        logger.info(msg)

def model_info(m, *, info_dict=None):
    '''If 'info_dict' is True, put JSON-able values into the given dictionary'''
    if info_dict is None:
        disp = 'shown' if m.display else 'hidden'
        line = '#%s, %s, %s' % (m.id_string, m.name, disp)
        if m.triangles is not None:
            line += ', %d triangles' % len(m.triangles)
        npos = len(m.positions)
        if npos > 1:
            line += ', %d instances' % npos
        spos = m.selected_positions
        if spos is not None and spos.sum() > 0:
            line += ', %d selected instances' % spos.sum()
        return line
    info_dict['spec'] = '#' + m.id_string
    info_dict['name'] = m.name
    info_dict['shown'] = m.display
    info_dict['num triangles'] = 0 if m.triangles is None else len(m.triangles)
    info_dict['num instances'] = len(m.positions)
    spos = m.selected_positions
    info_dict['num selected instances'] = 0 if spos is None else spos.sum()

def bounds_description(bounds):
    if bounds is None:
        bdesc = 'no bounding box'
    else:
        bdesc = ('bounds %.3g,%.3g,%.3g to ' % tuple(bounds.xyz_min) +
                 '%.3g,%.3g,%.3g' % tuple(bounds.xyz_max))
    return bdesc

def structure_info(m, *, info_dict=None):
    from chimerax.atomic import Structure
    if info_dict is None:
        if not isinstance(m, Structure):
            return ''

        line = ('\n%d atoms, %d bonds, %d residues, %d chains (%s)'
                % (m.num_atoms, m.num_bonds, m.num_residues, m.num_chains,
                   ','.join(m.residues.unique_chain_ids)))
        ncs = m.num_coordsets
        if ncs > 1:
            line += ', %d coordsets' % ncs
        pmap = m.pbg_map
        if pmap:
            line += '\n' + ', '.join('%d %s' % (pbg.num_pseudobonds, name)
                                     for name, pbg in pmap.items())
        return line
    if not isinstance(m, Structure):
        return
    info_dict['num atoms'] = m.num_atoms
    info_dict['num_bonds'] = m.num_bonds
    info_dict['num residues'] = m.num_residues
    info_dict['chains'] = [c.chain_id for c in m.chains]
    info_dict['num coordsets'] = m.num_coordsets
    info_dict['pseudobond groups'] = pb_groups = []
    for name, pbg in m.pbg_map.items():
        pb_groups.append({ 'name': name, 'num pseudobonds': pbg.num_pseudobonds })

def pseudobond_group_info(m, *, info_dict=None):
    from chimerax.atomic import PseudobondGroup
    if info_dict is None:
        if isinstance(m, PseudobondGroup):
            line = ', %d pseudobonds' % m.num_pseudobonds
        else:
            line = ''
        return line
    if not isinstance(m, PseudobondGroup):
        return
    info_dict['num pseudobonds'] = m.num_pseudobonds

def volume_info(m, *, info_dict=None):

    from chimerax.map import Volume
    if info_dict is None:
        if not isinstance(m, Volume):
            return ''

        size = 'size %d,%d,%d' % tuple(m.data.size)
        s0,s1,s2 = m.region[2]
        step = ('step %d' % s0) if s1 == s0 and s2 == s0 else 'step %d,%d,%d' % (s0,s1,s2)
        sx,sy,sz = m.data.step
        vsize = ('voxel size %.5g' % sx) if sx == sy and sy == sz else ('voxel size %.5g,%.5g,%.5g'
            % (sx,sy,sz))
        if m.surface_shown:
            level = 'level ' + ', '.join(('%.4g' % s.level for s in m.surfaces))
        else:
            level = 'level/intensity ' + ', '.join(('%.4g (%.2f)' % tuple(l) for l in m.image_levels))
        line = ' %s, %s, %s, %s' % (size, step, vsize, level)
        ms = m.matrix_value_statistics()
        line += ', value range %.5g - %.5g' % (ms.minimum, ms.maximum)
        line += ', value type %s' % str(m.data.value_type)
        sym = m.data.symmetries
        line += ', %d symmetry operators' % (len(sym) if sym else 0)
        return line
    if not isinstance(m, Volume):
        return
    info_dict['size'] = list(m.data.size)
    info_dict['step'] = list(m.region[2])
    info_dict['voxel size'] = list(m.data.step)
    info_dict['surface levels'] = [s.level for s in m.surfaces]
    info_dict['image levels'] = [list(l) for l in m.image_levels]
    ms = m.matrix_value_statistics()
    info_dict['minimum value'] = ms.minimum
    info_dict['maximum value'] = ms.maximum
    info_dict['value type'] = str(m.data.value_type)
    sym = m.data.symmetries
    info_dict['num symmetry operators'] = len(sym) if sym else 0

# ==============================================================================
# Code for sending REST request and waiting for response in a separate thread

from chimerax.core.tasks import Task
from contextlib import contextmanager

@contextmanager
def closing(thing):
    try:
        yield thing
    finally:
        thing.close()


class RESTTransaction(Task):

    def __init__(self, notifier):
        super().__init__(notifier.session)
        self.notifier = notifier

    def run(self, url, msg):
        # in new thread
        from urllib.parse import urlencode
        from urllib.request import urlopen, URLError
        full_url = "%s?%s" % (url, urlencode([("chimerax_notification", msg)]))
        try:
            with closing(urlopen(full_url, timeout=30)) as f:
                # Discard response since we cannot handle an error anyway
                f.read()
        except URLError:
            self.terminate()

    def on_finish(self):
        # in main thread
        from chimerax.core.tasks import TaskState
        if self.state == TaskState.TERMINATED:
            logger = self.session.logger
            try:
                Notifier.Destroy(self.notifier)
                # If we already destroyed the notifier, the warning
                # will not be sent again
                logger.warning("%s notifier for %s failed; notifier removed" %
                               (self.notifier.what, self.notifier.url))
            except KeyError:
                pass


class Notifier:

    SupportedTypes = ["models", "selection"]
    # A TYPE is supported when both _create_TYPE_handler
    # and _destroy_TYPE_handler methods are defined

    def __init__(self, what, client_id, session, prefix, url):
        self.what = what
        self.client_id = client_id
        self.session = session
        self.prefix = prefix
        self.url = url
        try:
            c_func = getattr(self, "_create_%s_handler" % what)
        except AttributeError:
            from chimerax.core.errors import UserError
            raise UserError("unsupported notification type: %s" % what)
        self._handler = c_func()
        self._handler_suspended = True
        self._destroy_handler = getattr(self, "_destroy_%s_handler" % what)
        self.Create(self)

    def start(self):
        self._handler_suspended = False
        msg = "listening for %s" % self.what
        self.session.logger.info(msg)

    def stop(self):
        self._destroy_handler()
        self.Destroy(self)
        msg = "stopped listening for %s" % self.what
        self.session.logger.info(msg)

    def suspend(self):
        self._handler_suspended = True
        msg = "suspended listening for %s" % self.what
        self.session.logger.info(msg)

    def resume(self):
        self._handler_suspended = False
        msg = "resumed listening for %s" % self.what
        self.session.logger.info(msg)

    def _notify(self, msgs):
        if self._handler_suspended:
            return
        if self.url is not None:
            # Notify via REST in a separate thread
            RESTTransaction(self).start(self.url, ''.join(msgs))
        else:
            # Just regular info messages
            logger = self.session.logger
            for msg in msgs:
                logger.info(msg)

    #
    # Methods for "models" notifications
    #
    def _create_models_handler(self):
        from chimerax.core.models import ADD_MODELS, REMOVE_MODELS
        add_handler = self.session.triggers.add_handler(ADD_MODELS,
                                                        self._notify_models)
        rem_handler = self.session.triggers.add_handler(REMOVE_MODELS,
                                                        self._notify_models)
        return [add_handler, rem_handler]

    def _destroy_models_handler(self):
        self.session.triggers.remove_handler(self._handler[0])
        self.session.triggers.remove_handler(self._handler[1])

    def _notify_models(self, trigger, trigger_data):
        msgs = []
        for m in trigger_data:
            msgs.append("%smodel %s" % (self.prefix, spec(m)))
        self._notify(msgs)

    #
    # Methods for "selection" notifications
    #
    def _create_selection_handler(self):
        from chimerax.core.selection import SELECTION_CHANGED
        handler = self.session.triggers.add_handler(SELECTION_CHANGED,
                                                    self._notify_selection)
        return handler

    def _destroy_selection_handler(self):
        self.session.triggers.remove_handler(self._handler)

    def _notify_selection(self, trigger, trigger_data):
        self._notify(["%sselection changed" % self.prefix])

    #
    # Class variables and methods for tracking by client identifier
    #

    _client_map = {}

    @classmethod
    def Find(cls, what, client_id, session=None, prefix=None, url=None):
        try:
            return cls._client_map[(what, client_id)]
        except KeyError:
            if session is None:
                from chimerax.core.errors import UserError
                raise UserError("using undefined notification client id: %s" %
                                client_id)
            return cls(what, client_id, session, prefix, url)

    @classmethod
    def Create(cls, n):
        key = (n.what, n.client_id)
        if key in cls._client_map:
            from chimerax.core.errors import UserError
            raise UserError("notification client id already in use: %s" %
                            n.client_id)
        else:
            cls._client_map[key] = n

    @classmethod
    def Destroy(cls, n):
        try:
            del cls._client_map[(n.what, n.client_id)]
        except KeyError:
            from chimerax.core.errors import UserError
            raise UserError("destroying undefined notification client id: %s" %
                            n.client_id)


# ==============================================================================
# Code for 'info shown' command

def get_shown_info(session, models=None):
    '''
    Returns a list of dictionaries, one per visible model, describing what is displayed.
    
    Only visible models are included in the output - hidden models are omitted entirely.
    Absence from the output means the model is not visible.
    '''
    from chimerax.atomic import AtomicStructure, Structure
    from chimerax.map import Volume
    from chimerax.atomic import PseudobondGroup
    
    if models is None:
        models = session.models.list()
    
    result = []
    
    for m in sorted(models, key=lambda m: m.id):
        # Use m.visible (not m.display) to check actual visibility.
        # A model is visible only if its own display is True AND all its
        # parents are also visible. This correctly filters out child models
        # whose parents are hidden.
        # Skip hidden models entirely - they're not shown, so don't report them.
        if not m.visible:
            continue
        
        model_info = {
            'id': '#' + m.id_string,
            'name': m.name,
            'type': type(m).__name__,
        }
        
        if isinstance(m, AtomicStructure) or isinstance(m, Structure):
            model_info['type'] = 'AtomicStructure' if isinstance(m, AtomicStructure) else 'Structure'
            _add_structure_display_info(session, m, model_info)
        elif isinstance(m, Volume):
            model_info['type'] = 'Volume'
            _add_volume_display_info(m, model_info)
        elif isinstance(m, PseudobondGroup):
            model_info['type'] = 'PseudobondGroup'
            _add_pseudobond_group_display_info(session, m, model_info)
        
        result.append(model_info)
    
    return result


def _add_structure_display_info(session, structure, info):
    '''Add display state info for an AtomicStructure or Structure.
    
    Only includes elements that are currently displayed.
    '''
    from chimerax.atomic import Residue, concise_residue_spec
    from chimerax.atomic.molsurf import MolecularSurface
    
    atoms = structure.atoms
    
    # Process each polymer chain - only include chains with something displayed
    chains_info = []
    for chain in structure.chains:
        chain_id = chain.chain_id
        chain_residues = chain.existing_residues
        if len(chain_residues) == 0:
            continue
        
        chain_atoms = chain_residues.atoms
        chain_info = {'id': chain_id}
        has_display = False
        
        # Atom display info - only include if atoms are displayed
        displayed_atoms = chain_atoms.filter(chain_atoms.displays)
        if len(displayed_atoms) > 0:
            has_display = True
            displayed_res = displayed_atoms.unique_residues
            chain_info['atoms'] = {'spec': concise_residue_spec(session, displayed_res)}
        
        # Ribbon display info - only include if ribbons are displayed
        ribbon_displays = chain_residues.ribbon_displays
        displayed_ribbon_count = ribbon_displays.sum()
        if displayed_ribbon_count > 0:
            has_display = True
            displayed_ribbon_res = chain_residues.filter(ribbon_displays)
            chain_info['ribbons'] = {'spec': concise_residue_spec(session, displayed_ribbon_res)}
        
        # Only add chain if something is displayed
        if has_display:
            chain_info['polymer_type'] = ('protein' if chain.polymer_type == Residue.PT_AMINO else 
                                          ('nucleic' if chain.polymer_type == Residue.PT_NUCLEIC else 'other'))
            chains_info.append(chain_info)
    
    if chains_info:
        info['chains'] = chains_info
    
    # Ligands - only include those with displayed atoms
    ligand_atoms = atoms.filter(atoms.structure_categories == "ligand")
    if len(ligand_atoms) > 0:
        ligands_info = []
        ligand_residues = ligand_atoms.unique_residues
        for res in ligand_residues:
            res_atoms = res.atoms
            displayed_count = res_atoms.displays.sum()
            if displayed_count > 0:
                lig_info = {
                    'name': res.name,
                    'chain': res.chain_id,
                    'number': res.number,
                    'spec': res.string(style='command', omit_structure=False),
                }
                if res.insertion_code:
                    lig_info['insertion_code'] = res.insertion_code
                # Only note partial display if not all atoms shown
                if displayed_count < len(res_atoms):
                    lig_info['atoms_shown'] = f"{int(displayed_count)}/{len(res_atoms)}"
                ligands_info.append(lig_info)
        if ligands_info:
            info['ligands'] = ligands_info
    
    # Ions - only include displayed ones
    ion_atoms = atoms.filter(atoms.structure_categories == "ions")
    if len(ion_atoms) > 0:
        ions_info = []
        ion_residues = ion_atoms.unique_residues
        for res in ion_residues:
            res_atoms = res.atoms
            if res_atoms.displays.sum() > 0:
                ion_info = {
                    'name': res.name,
                    'chain': res.chain_id,
                    'number': res.number,
                    'spec': res.string(style='command', omit_structure=False),
                }
                if res.insertion_code:
                    ion_info['insertion_code'] = res.insertion_code
                ions_info.append(ion_info)
        if ions_info:
            info['ions'] = ions_info
    
    # Solvent - only include if any is displayed
    solvent_atoms = atoms.filter(atoms.structure_categories == "solvent")
    displayed_solvent = solvent_atoms.filter(solvent_atoms.displays)
    if len(displayed_solvent) > 0:
        displayed_solvent_res = displayed_solvent.unique_residues
        info['solvent'] = {'spec': concise_residue_spec(session, displayed_solvent_res)}
    
    # Surfaces - only include displayed surfaces
    surfaces_info = []
    for surf in structure.surfaces():
        if isinstance(surf, MolecularSurface) and surf.display:
            show_atoms = surf.show_atoms
            if len(show_atoms) > 0:
                surf_info = {
                    'id': '#' + surf.id_string,
                    'name': surf.name,
                }
                show_res = show_atoms.unique_residues
                surf_info['spec'] = concise_residue_spec(session, show_res)
                surfaces_info.append(surf_info)
    if surfaces_info:
        info['surfaces'] = surfaces_info
    
    # Pseudobonds - only include displayed groups with displayed pseudobonds
    pbonds_info = []
    for name, pbg in structure.pbg_map.items():
        if pbg.display:
            pbs = pbg.pseudobonds
            displayed_pbs = pbs.filter(pbs.displays)
            if len(displayed_pbs) > 0:
                pb_info = {'name': name}
                if len(displayed_pbs) < len(pbs):
                    pb_info['count'] = f"{len(displayed_pbs)}/{len(pbs)}"
                pbonds_info.append(pb_info)
    if pbonds_info:
        info['pseudobonds'] = pbonds_info


def _add_volume_display_info(volume, info):
    '''Add display state info for a Volume.
    
    Only includes rendering modes that are currently active.
    '''
    if volume.surface_shown:
        info['surface_levels'] = [s.level for s in volume.surfaces]
    if volume.image_shown:
        info['image_levels'] = [list(l) for l in volume.image_levels]


def _add_pseudobond_group_display_info(session, pbg, info):
    '''Add display state info for a standalone PseudobondGroup.
    
    Only includes if pseudobonds are displayed.
    '''
    pbs = pbg.pseudobonds
    displayed_pbs = pbs.filter(pbs.displays)
    if len(displayed_pbs) > 0:
        if len(displayed_pbs) < len(pbs):
            info['pseudobonds'] = f"{len(displayed_pbs)}/{len(pbs)}"
        else:
            info['pseudobonds'] = len(displayed_pbs)
