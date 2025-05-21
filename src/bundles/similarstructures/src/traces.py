# vim: set expandtab ts=4 sw=4:

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

def similar_structures_traces(session, align_with = None, alignment_cutoff_distance = None,
                              show = 'close', distance = 4.0, max_segment_distance = 10.0,
                              min_segment_residues = 5, break_segment_distance = 5.0,
                              from_set = None, of_structures = None, replace = True):
    from .simstruct import similar_structure_results
    results = similar_structure_results(session, from_set)
    hits = results.named_hits(of_structures)

    if not results.have_c_alpha_coordinates():
        from . import coords
        if not coords.similar_structures_fetch_coordinates(session, ask = True, from_set = from_set,
                                                           of_structures = of_structures):
            return

    if alignment_cutoff_distance is None:
        alignment_cutoff_distance = results.alignment_cutoff_distance


    qchain = results.query_chain
    if qchain is None:
        from chimerax.core.errors import UserError
        raise UserError('Cannot position traces without query structure')

    qres = results.query_residues
    qatoms = qres.find_existing_atoms('CA')
    query_xyz = qatoms.coords
    if align_with is not None:
        ai = set(qatoms.indices(align_with.find_existing_atoms('CA')))
        ai.discard(-1)
        if len(ai) < 3:
            from chimerax.core.errors import UserError
            raise UserError('Similar structure traces align_with option specifies fewer than 3 aligned query atoms')

    chain_info = []
    from .simstruct import hit_coords, align_xyz_transform, hit_coordinates_sequence
    for hit in hits:
        hit_xyz = hit_coords(hit)
        hi, qi = results.hit_residue_pairing(hit)  # Coordinate array indices
        hxyz = hit_xyz[hi]
        qxyz = query_xyz[qi]
        if align_with is None:
            ahxyz, aqxyz = hxyz, qxyz
        else:
            from numpy import array
            mask = array([(i in ai) for i in qi], bool)
            ahxyz = hxyz[mask,:]
            aqxyz = qxyz[mask,:]
        if len(ahxyz) < 3:
                continue	# Not enough atoms to align.
        p, rms, npairs = align_xyz_transform(ahxyz, aqxyz, cutoff_distance=alignment_cutoff_distance)
        hxyz_aligned = p.transform_points(hxyz)

        hit_name = hit['database_full_id']
        qresnums = [qatoms[i].residue.number for i in qi]
        hitseq = hit_coordinates_sequence(hit)
        hresname = [hitseq[i] for i in hi]
        if show == 'close':
            rshow = _close_residues(hxyz_aligned, qxyz, distance, max_segment_distance,
                                    min_segment_residues, break_segment_distance)
        else:
            rshow = [True] * len(hresname)
        chain_info.append((hit_name, tuple(zip(hresname, qresnums, hxyz_aligned, rshow))))

    if len(chain_info) == 0:
        return None	# No hits had enough alignment atoms.

    hit_names = [hit['database_full_id'] for hit in hits]
    if replace:
        tmodels = _existing_trace_models(session, results, hit_names)
        if tmodels:
            session.models.close(tmodels)

    name = f'{results.name} traces'
    model = BackboneTraces(name, session, chain_info, similar_structures_id = results.name)

    model.hit_names = hit_names
    model.position = qchain.structure.scene_position
    session.models.add([model])

    msg = f'{model.num_chains} backbone traces'
    session.logger.info(msg)
    return model

def _close_residues(hxyz_aligned, qxyz, distance, max_segment_distance, min_segment_residues,
                    break_segment_distance):
    # Figure which distant residues to hide initially 
    fragments = _distant_c_alpha_fragments(hxyz_aligned, break_segment_distance)
    if distance is not None and distance > 0:
        cfrags = []
        for start,end in fragments:
            cfrags.extend(_close_fragments(hxyz_aligned[start:end], qxyz[start:end],
                                           distance, max_segment_distance, start))
        fragments = cfrags
    min_res = max(2, min_segment_residues)
    fragments = [(s,e) for s,e in fragments if e-s >= min_res]
    from numpy import zeros
    rshow = zeros((len(hxyz_aligned),), bool)
    for s,e in fragments:
        rshow[s:e] = True
    return rshow

def _distant_c_alpha_fragments(hxyz, break_distance = 5):
    d = hxyz[1:,:] - hxyz[:-1,:]
    d2 = (d*d).sum(axis = 1)
    breaks = (d2 > break_distance*break_distance).nonzero()[0]
    fragments = []
    start = 0
    for b in breaks:
        fragments.append((start, b+1))
        start = b+1
    fragments.append((start, len(hxyz)))
    return fragments

def _close_fragments(xyz, ref_xyz, distance, max_segment_distance = None, offset = 0):
    d = xyz - ref_xyz
    d2 = (d*d).sum(axis = 1)
    mask = (d2 <= distance*distance)
    if max_segment_distance is not None:
        n = len(xyz)
        max_dist2 = max_segment_distance * max_segment_distance
        for start, end in _mask_intervals(~mask):
            if start > 0 and end < n and d2[start:end].max() <= max_dist2:
                mask[start:end] = True  # Keep interior interval if largest distance is not too large.
    fragments = _mask_intervals(mask)
    if offset != 0:
        fragments = [(start+offset, end+offset) for start, end in fragments]
    return fragments

def _mask_intervals(mask):
    ends = list((mask[1:] != mask[:-1]).nonzero()[0]+1)
    if mask[0]:
        ends.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    return tuple(zip(ends[0::2], ends[1::2]))

def _existing_trace_models(session, results, hit_names):
    from .simstruct import similar_structure_results
    tmodels = [m for m in session.models.list(type = BackboneTraces)
               if similar_structure_results(session, m.similar_structures_id, raise_error = False) is results
               and m.hit_names == hit_names]
    return tmodels

aa_1_to_3 = {'C':'CYS', 'D':'ASP', 'S':'SER', 'Q':'GLN', 'K':'LYS',
             'I':'ILE', 'P':'PRO', 'T':'THR', 'F':'PHE', 'N':'ASN', 
             'G':'GLY', 'H':'HIS', 'L':'LEU', 'R':'ARG', 'W':'TRP', 
             'A':'ALA', 'V':'VAL', 'E':'GLU', 'Y':'TYR', 'M':'MET'}

from chimerax.atomic import AtomicStructure
class BackboneTraces(AtomicStructure):
    def __init__(self, name, session, chain_info, similar_structures_id):
        '''chains is a list of (chain_id, c_alphas) where c_alphas is a list of (res_code, res_num, (x,y,z), show).'''
        self.similar_structures_id = similar_structures_id
        AtomicStructure.__init__(self, session, name = name, auto_style = False, log_info = False)
        if chain_info:
            self._create_chains(chain_info)
        self._remember_visible_residues()
        register_context_menu()  # Register select mouse mode double click context menu

    def _create_chains(self, chain_info, ribbon_color = (180,180,180,255)):
        pbg = self.pseudobond_group('missing structure')
        for chain_id, c_alphas in chain_info:
            aprev = None
            for res_code, res_num, xyz, show in c_alphas:
                res_name = aa_1_to_3.get(res_code, 'UNK')
                r = self.new_residue(res_name, chain_id, res_num)
                r.ribbon_display = show
                r.ribbon_color = ribbon_color
                a = self.new_atom('CA', 'C')
                a.coord = xyz
                a.display = False	# Don't show atoms
                r.add_atom(a)
                if aprev:
                    pbg.new_pseudobond(aprev, a)
                aprev = a

    def show_traces(self, names, show = True, other = False):
        names_set = set(names)
        for chain in self.chains:
            name = chain.chain_id
            change = (name not in names_set) if other else (name in names_set)
            if change:
                self._show_visible_residues(chain.existing_residues, show)

    def show_all_traces(self):
        self._show_visible_residues(self.residues)

    def _show_visible_residues(self, residues, show = True):
        residues.ribbon_displays = tuple(r._show_in_trace for r in residues) if show else False

    def _remember_visible_residues(self):
        from chimerax.atomic import Residue
        Residue.register_attr(self.session, '_show_in_trace', 'Similar Structures', attr_type = bool)
        for r in self.residues:
            r._show_in_trace = r.ribbon_display

    def show_full_traces(self):
        for c in self.chains:
            res = c.existing_residues
            if res.ribbon_displays.any():
                res.ribbon_displays = True

    def show_only_close_traces(self):
        for c in self.chains:
            res = c.existing_residues
            if res.ribbon_displays.any():
                self._show_visible_residues(res)

    def take_snapshot(self, session, flags):
        as_data = AtomicStructure.take_snapshot(self, session, flags)
        data = {
            'atomic structure': as_data,
            'similar_structures_id': self.similar_structures_id,
            'version': 1
        }
        return data

    @staticmethod
    def restore_snapshot(session, data):
        s = BackboneTraces('Backbone traces', session, [], data['similar_structures_id'])
        AtomicStructure.set_state_from_snapshot(s, session, data['atomic structure'])
        return s
    
# Add hide and delete atoms/bonds/pseudobonds to double-click selection context menu
from chimerax.mouse_modes import SelectContextMenuAction
class BackboneTraceMenuEntry(SelectContextMenuAction):
    def __init__(self, action, menu_text):
        self.action = action
        self.menu_text = menu_text
    def label(self, session):
        hit_name = self._hit_name(session)[0]
        label = self.menu_text
        if '%s' in label:
            label = label % hit_name
        return label
    def criteria(self, session):
        return self._hit_name(session)[0] is not None
    def callback(self, session):
        hit_name, bt_model = self._hit_name(session)
        if hit_name is None or bt_model is None:
            return
        ssid = bt_model.similar_structures_id
        from chimerax.core.commands import run
        a = self.action
        if a == 'open':
            run(session, f'similarstructures open {hit_name} from {ssid}')
        elif a == 'scroll to':
            run(session, f'similarstructures scrollto {hit_name} from {ssid}')
        elif a == 'show only':
            bt_model.show_traces([hit_name])
            bt_model.show_traces([hit_name], show=False, other=True)
        elif a == 'show all':
            bt_model.show_all_traces()
        elif a == 'show full':
            bt_model.show_full_traces()
        elif a == 'show close':
            bt_model.show_only_close_traces()
    def _hit_name(self, session):
        from chimerax.atomic import selected_atoms
        atoms = selected_atoms(session)
        if len(atoms) == 1:
            a = atoms[0]
            bt_model = a.structure
            if isinstance(bt_model, BackboneTraces):
                hit_name = a.residue.chain.chain_id
                return hit_name, bt_model
        return None, None
    
_registered_context_menu = False
def register_context_menu():
    global _registered_context_menu
    if not _registered_context_menu:
        from chimerax.mouse_modes import SelectMouseMode
        SelectMouseMode.register_menu_entry(BackboneTraceMenuEntry('open', 'Open similar structure %s'))
        SelectMouseMode.register_menu_entry(BackboneTraceMenuEntry('scroll to', 'Show %s in similar structures table'))
        SelectMouseMode.register_menu_entry(BackboneTraceMenuEntry('show only', 'Show only trace %s'))
        SelectMouseMode.register_menu_entry(BackboneTraceMenuEntry('show all', 'Show all traces'))
        SelectMouseMode.register_menu_entry(BackboneTraceMenuEntry('show full', 'Show full traces'))
        SelectMouseMode.register_menu_entry(BackboneTraceMenuEntry('show close', 'Show only close traces'))
        _registered_context_menu = True
    
def register_similar_structures_traces_command(logger):
    from chimerax.core.commands import CmdDesc, register, FloatArg, IntArg, BoolArg, StringArg, EnumOf
    from chimerax.atomic import ResiduesArg
    desc = CmdDesc(
        required = [],
        keyword = [('align_with', ResiduesArg),
                   ('alignment_cutoff_distance', FloatArg),
                   ('show', EnumOf(['all','close'])),
                   ('distance', FloatArg),
                   ('max_segment_distance', FloatArg),
                   ('min_segment_residues', IntArg),
                   ('break_segment_distance', FloatArg),
                   ('from_set', StringArg),
                   ('of_structures', StringArg),
                   ('replace', BoolArg),
                   ],
        synopsis = 'Show backbone traces of similar structures aligned to query structure.'
    )
    register('similarstructures traces', desc, similar_structures_traces, logger=logger)
