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

# -----------------------------------------------------------------------------
#
def combine_geometry_vtp(geom):
    vc = tc = 0
    for va, ta, pos in geom:
        n, nv, nt = len(pos), len(va), len(ta)
        vc += n*nv
        tc += n*nt

    from numpy import empty, float32, int32
    varray = empty((vc,3), float32)
    tarray = empty((tc,3), int32)

    v = t = 0
    for va, ta, pos in geom:
        n, nv, nt = len(pos), len(va), len(ta)
        for p in pos:
            varray[v:v+nv,:] = va if p.is_identity() else p*va
            tarray[t:t+nt,:] = ta
            tarray[t:t+nt,:] += v
            v += nv
            t += nt
    
    return varray, tarray

# -----------------------------------------------------------------------------
#
def combine_geometry_vnt(geoms):
    from numpy import concatenate
    cva = concatenate([va for va,na,ta in geoms])
    cna = concatenate([na for va,na,ta in geoms])
    cta = concatenate([ta for va,na,ta in geoms])
    voffset = 0
    toffset = 0
    for va,na,ta in geoms:
        cta[toffset:toffset+len(ta)] += voffset
        voffset += len(va)
        toffset += len(ta)
    return cva,cna,cta

# -----------------------------------------------------------------------------
#
def combine_geometry_vntc(geoms):
    from numpy import concatenate
    cva = concatenate([va for va,na,ta,ca in geoms])
    cna = concatenate([na for va,na,ta,ca in geoms])
    cta = concatenate([ta for va,na,ta,ca in geoms])
    cca = concatenate([ca for va,na,ta,ca in geoms])
    voffset = 0
    toffset = 0
    for va,na,ta,ca in geoms:
        cta[toffset:toffset+len(ta)] += voffset
        voffset += len(va)
        toffset += len(ta)
    return cva,cna,cta,cca

# -----------------------------------------------------------------------------
#
def combine_geometry_xvnt(surfs):
    nv = sum(len(va) for extra, va, na, ta in surfs)
    from numpy import empty, float32, uint8, concatenate
    cva = empty((nv,3), float32)
    cna = empty((nv,3), float32)
    voffset = 0
    tlist = []
    for extra, va, na, ta in surfs:
        snv = len(va)
        cva[voffset:voffset+snv,:] = va
        cna[voffset:voffset+snv,:] = na
        ta += voffset
        tlist.append(ta)
        voffset += snv
    cta = concatenate(tlist)
    return cva, cna, cta

# -----------------------------------------------------------------------------
#
def combine_geometry_xvntctp(geom):
    vc = tc = 0
    tex_coord = False
    for extra, va, na, tca, ta, pos in geom:
        n, nv, nt = len(pos), len(va), len(ta)
        vc += n*nv
        tc += n*nt
        if tca is not None:
            tex_coord = True
        elif tex_coord:
            raise RuntimeError('Cannot combine some models with texture coordinates'
                               ' and others without texture coordinates')

    from numpy import empty, float32, int32
    varray = empty((vc,3), float32)
    narray = empty((vc,3), float32)
    tcarray = empty((vc,2), float32) if tex_coord else None
    tarray = empty((tc,3), int32)

    v = t = 0
    for extra, va, na, tca, ta, pos in geom:
        n, nv, nt = len(pos), len(va), len(ta)
        for p in pos:
            varray[v:v+nv,:] = va if p.is_identity() else p*va
            narray[v:v+nv,:] = na if p.is_identity() else p.transform_vectors(na)
            if tex_coord:
                tcarray[v:v+nv,:] = tca
            tarray[t:t+nt,:] = ta
            tarray[t:t+nt,:] += v
            v += nv
            t += nt
    
    return varray, narray, tcarray, tarray

# -----------------------------------------------------------------------------
#
def combine_geometry_vte(geom):
    nv = nt = 0
    for va, ta, ema in geom:
        nv += len(va)
        nt += len(ta)
    from numpy import empty, float32, int32, uint8
    cva = empty((nv,3), float32)
    cta = empty((nt,3), int32)
    cea = empty((nt,), uint8)
    voffset = 0
    toffset = 0
    for va, ta, ema in geom:
        vc, tc = len(va), len(ta)
        cva[voffset:voffset+vc,:] = va
        cta[toffset:toffset+tc,:] = ta
        cta[toffset:toffset+tc,:] += voffset
        cea[toffset:toffset+tc] = ema
        voffset += vc
        toffset += tc
    return cva, cta, cea

# -----------------------------------------------------------------------------
#
def combine_geometry(geom, vertex_position = 0, triangle_position = 1):
    '''
    Combine triangle geometry given by multiple pairs of vertex and triangle arrays.
    The arrays are concatenated into one array of vertices and one array of triangles
    with the triangle array vertex indices offset for use with the concatenated vertex array.
    '''
    if len(geom) <= 1:
        return geom

    # Concatenate arrays
    na = len(geom[0])
    accum = [[] for a in range(na)]
    for g in geom:
        for i,a in enumerate(g):
            accum[i].append(a)
    from numpy import concatenate
    cgeom = [concatenate(alist) for alist in accum]

    # Adjust triangle offsets
    voffset = 0
    toffset = 0
    cta = cgeom[triangle_position]
    for g in geom:
        vc, tc = len(g[vertex_position]), len(g[triangle_position])
        cta[toffset:toffset+tc,:] += voffset
        voffset += vc
        toffset += tc

    return cgeom
