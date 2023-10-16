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

def read_positions(session, stream, name, models=None, match_names=False, child_models=False):
    """
    Read position matrices and set positions of specified models.
    """

    from chimerax.core.errors import UserError
    if models is None or len(models) == 0:
        raise UserError('Must specify models when opening positions file %s.' % stream.name)

    if not child_models:
        models = _exclude_child_models(models)
    
    lines = stream.readlines()
    places = _parse_places(lines, stream.name)
    if len(places) == 0:
        raise UserError('No positions specified in positions file %s.' % stream.name)

    if match_names:
        n2p = dict(places)
        mpos = [(m, n2p[m.name]) for m in models if m.name in n2p]
        for m, position in mpos:
            m.position = position
        msg = (('Set position of model %s' % m.name)
               if len(mpos) == 1 else
               ('Set %d positions' % len(mpos)))
    elif len(models) == 1:
        m = models[0]
        from chimerax.geometry import Places
        m.positions = Places(p for name,p in places)
        msg = (('Set position of model %s' % m.name)
               if len(places) == 1 else
               ('Set %d position for model %s' % (len(places), m.name)))
    elif len(places) != len(models):
        raise UserError('Number of models (%d) does not equal number of positions (%d) in position file %s'
                        % (len(models), len(places), stream.name))
    else:
        for m, (name, position) in zip(models, places):
            m.position = position
        msg = 'Set %d positions' % len(places)

    return [], msg
    
# ----------------------------------------------------------------------------------
#
def _parse_places(lines, filename):
    places = []
    from chimerax.core.errors import UserError
    for line_num, line in enumerate(lines):
        if line.startswith('#'):
            continue	# Comment line
        fields = line.split(',')
        if len(fields) == 1 and line.strip() == '':
            continue	# Allow blank lines
        if len(fields) < 13:
            raise UserError('Expected 13 fields in line %d of file %s, got %d:\n"%s"'
                            % (line_num+1, filename, len(fields), line))
        name = fields[0].strip()
        try:
            mvalues = [float(x) for x in fields[1:]]
        except ValueError:
            raise UserError('Bad number format in line %d of file %s:\n"%s"'
                            % (line_num+1, stream.name, line))
        from numpy import array, float64
        matrix = array(mvalues, float64).reshape((3,4))
        from chimerax.geometry import Place
        p = Place(matrix = matrix)
        places.append((name, p))

    return places
    
# ----------------------------------------------------------------------------------
#
def save_positions(session, path, models=None, child_models=False):
    """
    Save position matrices of specified models.
    """

    from chimerax.core.errors import UserError
    if models is None or len(models) == 0:
        raise UserError('Must specify models when saving positions file %s.' % path)

    if not child_models:
        models = _exclude_child_models(models)
        
    lines = []
    for m in models:
        matrix = m.position.matrix
        mstring = ','.join('%.12g' % v for v in matrix.reshape((12,)))
        lines.append('%s,%s' % (m.name, mstring))
        
    f = open(path, 'w')
    f.write('\n'.join(lines))
    f.close()

# ----------------------------------------------------------------------------------
#
def _exclude_child_models(models):
    mset = set(models)
    return [m for m in models if m.parent not in mset]
        
