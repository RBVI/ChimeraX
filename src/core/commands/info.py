# vim: set expandtab shiftwidth=4 softtabstop=4:


def info(session, models=None):
    '''
    Report state of models, such as whether they are displayed, color, number of children,
    number of instances...

    Parameters
    ----------
    models : list of models
    '''
    m = session.models
    if models is None:
        models = m.list()
    
    lines = []
    msort = list(models)
    msort.sort(key = lambda m: m.id)
    for m in msort:
        disp = 'shown' if m.display else 'hidden'
        line = '#%s, %s, %s' % (m.id_string(), m.name, disp)
        if m.triangles is not None:
            line += ', %d triangles' % len(m.triangles)
        b = m.bounds()
        if b is None:
            line += ', no bounding box'
        else:
            line += ', bounds %.3g,%.3g,%.3g to ' % tuple(b.xyz_min) + '%.3g,%.3g,%.3g' % tuple(b.xyz_max)
        npos = len(m.positions)
        if npos > 1:
            line += ', %d instances' % npos
        spos = m.selected_positions
        if spos is not None and spos.sum() > 0:
            line += ', %d selected instances' % spos.sum()
        from ..atomic import Structure
        if isinstance(m, Structure):
            line += ('\n%d atoms, %d bonds, %d residues, %d chains'
                    % (m.num_atoms, m.num_bonds, m.num_residues, m.num_chains))
            pmap = m.pbg_map
            if pmap:
                line += '\n' + ', '.join('%d %s' % (pbg.num_pseudobonds, name)
                                         for name, pbg in pmap.items())
        from ..map import Volume
        if isinstance(m, Volume):
            size = 'size %d,%d,%d' % tuple(m.data.size)
            s0,s1,s2 = m.region[2]
            step = ('step %d' % s0) if s1 == s0 and s2 == s0 else 'step %d,%d,%d' % (s0,s1,s2)
            if m.representation == 'surface':
                level = 'level ' + ', '.join(('%.4g' % l for l in m.surface_levels))
            else:
                level = 'level/intensity ' + ', '.join(('%.4g (%.2f)' % tuple(l) for l in m.solid_levels))
            line += ' %s, %s, %s' % (size, step, level)
            ms = m.matrix_value_statistics()
            line += ', value range %.5g - %.5g' % (ms.minimum, ms.maximum)
            line += ', value type %s' % str(m.data.value_type)
            line += ', %d symmetry operators' % len(m.data.symmetries)
        lines.append(line)
    msg = '%d models\n' % len(models) + '\n'.join(lines)
    session.logger.info(msg)

def register_command(session):
    from . import CmdDesc, register, ModelsArg
    desc = CmdDesc(optional=[('models', ModelsArg)],
                   synopsis='report info about models')
    register('info', desc, info)
