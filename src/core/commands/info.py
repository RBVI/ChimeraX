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
        from ..atomic import AtomicStructure
        if isinstance(m, AtomicStructure):
            line += ('\n%d atoms, %d bonds, %d residues, %d chains'
                    % (m.num_atoms, m.num_bonds, m.num_residues, m.num_chains))
            pmap = m.pbg_map
            if pmap:
                line += '\n' + ', '.join('%d %s' % (pbg.num_pseudobonds, name)
                                         for name, pbg in pmap.items())
                           
        lines.append(line)
    msg = '%d models\n' % len(models) + '\n'.join(lines)
    session.logger.info(msg)

def register_command(session):
    from . import CmdDesc, register, ModelsArg
    desc = CmdDesc(optional=[('models', ModelsArg)],
                   synopsis='report info about models')
    register('info', desc, info)
