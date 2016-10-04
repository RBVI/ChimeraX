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

#
# Smooth lines of a drawing.
#
def smoothlines(session, models, step_factor = 0.1, iterations = 10, replace = False):

    for model in models:
        ta = model.triangles
        if ta.shape[1] != 2:
            from chimerax.core.errors import UserError
            raise UserError('Model %s does not contain lines' % model.name)

        va = model.vertices
        sva = smoothed_vertices(va, ta, step_factor, iterations)

        if replace:
            model.vertices = sva
        else:
            from chimerax.core.models import Model
            m = Model('%s smoothed' % model.name, session)
            m.vertices = sva
            m.triangles = ta
            m.normals = model.normals
            m.display_style = m.Mesh
            m.color = model.color
            session.models.add([m])
            model.display = False

def register_smoothlines_command():

    from chimerax.core.commands import CmdDesc, register, ModelsArg, FloatArg, IntArg, BoolArg

    desc = CmdDesc(
        required = [('models', ModelsArg)],
        keyword = [('step_factor', FloatArg),
                   ('iterations', IntArg),
                   ('replace', BoolArg)]
    )
    register('smoothlines', desc, smoothlines)

def smoothed_vertices(vertices, lines, step_factor, iterations):
    n = len(vertices)
    from numpy import ones, int32
    c = ones((n,), int32)
    v0, v1 = lines[:,0], lines[:,1]
    c[v0] += 1
    c[v1] += 1
    va = vertices.copy()
    va0 = vertices.copy()
    for i in range(iterations):
        va[v0,:] += va0[v1,:]
        va[v1,:] += va0[v0,:]
        va /= c.reshape((n,1))
        # va0 = va*f + va0*(1-f), f = step_factor
        va *= step_factor
        va0 *= 1-step_factor
        va0 += va
        va[:] = va0
        
    return va
