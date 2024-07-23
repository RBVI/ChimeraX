# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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

def foldseek_traces(session, align_with = None, cutoff_distance = None, close_only = 4.0,
                    tube = True, radius = 0.1, segment_subdivisions = 3, circle_subdivisions = 6):
    from .gui import foldseek_panel
    fp = foldseek_panel(session)
    if fp is None:
        return
    if cutoff_distance is None:
        cutoff_distance = fp.alignment_cutoff_distance
    _show_backbone_traces(session, fp.hits, fp.results_query_chain, align_with = align_with,
                          cutoff_distance = cutoff_distance, close_only = close_only,
                          tube = tube, radius = radius,
                          segment_subdivisions = segment_subdivisions,
                          circle_subdivisions = circle_subdivisions)

def _show_backbone_traces(session, hits, query_chain, align_with = None, cutoff_distance = 2.0, close_only = 4.0,
                          tube = True, radius = 0.1, segment_subdivisions = 3, circle_subdivisions = 6):
    traces = []
    from .foldseek import alignment_residues, hit_coords, hit_residue_pairing, align_xyz_transform
    qres = alignment_residues(query_chain.existing_residues)
    qatoms = qres.existing_principal_atoms
    query_xyz = qatoms.coords
    if align_with is not None:
        ai = set(qatoms.indices(align_with.existing_principal_atoms))
        ai.discard(-1)
        if len(ai) < 3:
            from chimerax.core.errors import UserError
            raise UserError('Foldseek traces align_with specifies fewer than 3 aligned query atoms')
    for hit in hits:
        hit_xyz = hit_coords(hit)
        hi, qi = hit_residue_pairing(hit)
        hxyz = hit_xyz[hi]
        qxyz = query_xyz[qi]
        if align_with is None:
            ahxyz, aqxyz = hxyz, qxyz
        else:
            from numpy import array
            mask = array([(i in ai) for i in qi], bool)
            if mask.sum() < 3:
                continue	# Not enough atoms to align.
            ahxyz = hxyz[mask,:]
            aqxyz = qxyz[mask,:]
        p, rms, npairs = align_xyz_transform(ahxyz, aqxyz, cutoff_distance=cutoff_distance)
        ahxyz = p.transform_points(hxyz)
        breaks = ((hi[1:] - hi[:-1]) > 1).nonzero()[0]
        if close_only is not None and close_only > 0:
            cmask = _close_mask(ahxyz, qxyz, close_only)
            ahxyz = ahxyz[cmask]
            breaks = _mask_breaks(cmask, breaks)
        trace_name = hit['database_full_id']
        traces.append(trace_name)
        if len(breaks) > 0:
            traces.extend(_break_chain(ahxyz, breaks))
        else:
            traces.append(ahxyz)
    if len(traces) == 0:
        return None	# No hits had enough alignment atoms.
    if tube:
        surf = _create_tube_traces_model(session, traces, radius = radius,
                                         segment_subdivisions = segment_subdivisions,
                                         circle_subdivisions = circle_subdivisions)
    else:
        surf = _create_line_traces_model(session, traces)

    surf.position = query_chain.structure.scene_position
    session.models.add([surf])
    print (f'{surf.trace_count} traces have {len(surf.vertices)} vertices')
    return surf

def _create_line_traces_model(session, traces):
    vertices, lines, names = _line_traces(traces)
    normals = None
    ft = FoldseekTraces('Foldseek traces', session)
    ft.set_geometry(vertices, normals, lines)
    ft.display_style = ft.Mesh
    ft.set_trace_names(names)
    return ft

def _line_traces(traces):
    traces_xyz = [t for t in traces if not isinstance(t, str)]	# Filter out labels
    from numpy import concatenate, float32, empty, int32, arange
    vertices = concatenate(traces_xyz, dtype = float32)
    nlines = sum([len(t)-1 for t in traces_xyz],0)
    lines = empty((nlines,2), int32)
    vp = offset = 0
    names = []
    tstart = []
    for t in traces:
        if isinstance(t, str):
            names.append(t)
            tstart.append(offset)
        else:
            tlen = len(t)
            tlines = lines[offset:offset+tlen-1,:]
            tlines[:,0] = tlines[:,1] = arange(vp, vp+tlen-1, dtype=int32)
            tlines[:,1] += 1
            vp += tlen
            offset += tlen-1
    return vertices, lines, (names, tstart)

def _create_tube_traces_model(session, traces, radius = 0.1, segment_subdivisions = 3, circle_subdivisions = 6):
    vertices, normals, triangles, names = _tube_traces(traces, radius = radius,
                                                       segment_subdivisions = segment_subdivisions,
                                                       circle_subdivisions = circle_subdivisions)
    ft = FoldseekTraces('Foldseek traces', session)
    ft.set_geometry(vertices, normals, triangles)
    ft.set_trace_names(names)
    return ft

def _tube_traces(traces, radius = 0.1, segment_subdivisions = 5, circle_subdivisions = 6):
    vnt = []
    names = []
    tstart = []
    tcount = 0
    from chimerax.surface.tube import tube_spline
    for trace in traces:
        if isinstance(trace, str):
            names.append(trace)
            tstart.append(tcount)
        else:
            v,n,t = tube_spline(trace, radius = radius,
                                segment_subdivisions = segment_subdivisions,
                                circle_subdivisions = circle_subdivisions)
            vnt.append((v,n,t))
            tcount += len(t)
            
    from chimerax.surface import combine_geometry_vnt
    vertices, normals, triangles = combine_geometry_vnt(vnt)

    return vertices, normals, triangles, (names, tstart)

def _close_mask(xyz, ref_xyz, distance):
    d = xyz - ref_xyz
    d2 = (d*d).sum(axis = 1)
    mask = (d2 <= distance*distance)
    return mask

def _break_chain(xyz, break_after_indices):
    pieces = []
    b0 = 0
    for b in break_after_indices:
        if b > b0:  # Exclude pieces of length 1
            pieces.append(xyz[b0:b+1,:])
        b0 = b+1
    return pieces

def _mask_breaks(mask, breaks):
    mbreaks = (mask[1:] != mask[:-1]).nonzero()[0]
    from numpy import concatenate, cumsum, unique
    abreaks = concatenate((breaks, mbreaks))
    reindex = cumsum(mask)-1
    ribreaks = reindex[abreaks]
    uribreaks = unique(ribreaks)
    max = reindex[-1]
    nbreaks = uribreaks[(uribreaks >= 0) & (uribreaks < max)]  # Remove breaks beyond ends
    return nbreaks

# Allow mouse hover to identify hits
from chimerax.core.models import Surface
class FoldseekTraces(Surface):
    def __init__(self, name, session):
        Surface.__init__(self, name, session)
        register_context_menu()  # Register select mouse mode double click context menu
    def set_trace_names(self, trace_names):
        self._trace_names, self._trace_start_triangle = trace_names
    @property
    def trace_count(self):
        return len(self._trace_names)
    def triangle_trace_name(self, triangle_number):
        from bisect import bisect_right
        i = bisect_right(self._trace_start_triangle, triangle_number) - 1
        trace_name = self._trace_names[i]
        return trace_name
    def select_trace(self, triangle_number):
        tst = self._trace_start_triangle
        from bisect import bisect_right
        i = bisect_right(tst, triangle_number) - 1
        nt = len(self.triangles)
        from numpy import zeros
        mask = zeros((nt,), bool)
        end = nt if i+1 == len(tst) else tst[i+1]
        mask[tst[i]:end] = True
        self.highlighted = False
        self.highlighted_triangles_mask = mask
    def first_intercept(self, mxyz1, mxyz2, exclude=None):
        pick = super().first_intercept(mxyz1, mxyz2, exclude=exclude)
        from chimerax.graphics import PickedTriangle
        if isinstance(pick, PickedModel) and hasattr(pick, 'picked_triangle'):
            pick = PickedFoldseekTrace(self, pick.picked_triangle.triangle_number, pick.distance)
        return pick
    @property
    def selected_hit(self):
        tmask = self.highlighted_triangles_mask
        if tmask is None or not tmask.any():
            return None
        tnum = tmask.nonzero()[0][0]	# Unfortunately numpy does not have an argfirst().
        return self.triangle_trace_name(tnum)

from chimerax.core.models import PickedModel
class PickedFoldseekTrace(PickedModel):
    def __init__(self, model, triangle_number, distance):
        super().__init__(model, distance)
        self.triangle_number = triangle_number
    def description(self):
        trace_name = self.model.triangle_trace_name(self.triangle_number)
        return f'Foldseek trace {trace_name}'
    def select(self, mode='add'):
        self.model.select_trace(self.triangle_number)
    def specifier(self):
        return None	# If model spec given then it overrides select() method.
    
# Add hide and delete atoms/bonds/pseudobonds to double-click selection context menu
from chimerax.mouse_modes import SelectContextMenuAction
class FoldseekHitMenuEntry(SelectContextMenuAction):
    def __init__(self, action, menu_text):
        self.action = action
        self.menu_text = menu_text
    def label(self, ses):
        hname = self._hit_name(ses)
        return self.menu_text % hname
    def criteria(self, ses):
        return self._hit_name(ses) is not None
    def _hit_name(self, session):
        for ft in session.models.list(type = FoldseekTraces):
            hname = ft.selected_hit
            if hname:
                return hname
        return None
    def callback(self, ses):
        hname = self._hit_name(ses)
        if hname:
            from chimerax.core.commands import run
            run(ses, f'foldseek {self.action} {hname}')
    
_registered_context_menu = False
def register_context_menu():
    global _registered_context_menu
    if not _registered_context_menu:
        from chimerax.mouse_modes import SelectMouseMode
        SelectMouseMode.register_menu_entry(FoldseekHitMenuEntry('open', 'Open Foldseek hit %s'))
        SelectMouseMode.register_menu_entry(FoldseekHitMenuEntry('show', 'Show %s in Foldseek results table'))
        _registered_context_menu = True
    
def register_foldseek_traces_command(logger):
    from chimerax.core.commands import CmdDesc, register, FloatArg, IntArg, BoolArg
    from chimerax.atomic import ResiduesArg
    desc = CmdDesc(
        required = [],
        keyword = [('align_with', ResiduesArg),
                   ('cutoff_distance', FloatArg),
                   ('close_only', FloatArg),
                   ('tube', BoolArg),
                   ('radius', FloatArg),
                   ('segment_subdivisions', IntArg),
                   ('circle_subdivisions', IntArg),
                   ],
        synopsis = 'Show backbone traces of Foldseek hits aligned to query structure.'
    )
    register('foldseek traces', desc, foldseek_traces, logger=logger)
