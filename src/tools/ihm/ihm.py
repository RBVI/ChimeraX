# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
ihm: Integrative Hybrid Model file format support
=================================================
"""
def read_ihm(session, filename, name, *args, **kw):
    """Read an integrative hybrid models file creating sphere models and restraint models

    :param filename: either the name of a file or a file-like object

    Extra arguments are ignored.
    """

    if hasattr(filename, 'read'):
        # Got a stream
        stream = filename
        filename = stream.name
        stream.close()

    table_names = ['ihm_sphere_obj_site', 'ihm_cross_link_restraint']
    from chimerax.core.atomic import mmcif
    spheres_obj_site, xlink_restraint = mmcif.get_mmcif_tables(filename, table_names)

    sos_fields = [
        'seq_id_begin',
        'seq_id_end',
        'asym_id',
        'cartn_x',
        'cartn_y',
        'cartn_z',
        'object_radius',
        'model_id']
    spheres = spheres_obj_site.fields(sos_fields)
    mspheres = {}
    for seq_beg, seq_end, asym_id, x, y, z, radius, model_id in spheres:
        sb, se = int(seq_beg), int(seq_end)
        xyz = float(x), float(y), float(z)
        r = float(radius)
        mid = int(model_id)
        mspheres.setdefault(mid, []).append((sb,se,asym_id,xyz,r))

    from os.path import basename, splitext
    name = splitext(basename(filename))[0]
    models = [IHMSphereModel(session, '%s %d' % (name,mid) , slist) for mid, slist in mspheres.items()]
    for m in models[1:]:
        m.display = False	# Only show first model.

    xlink_fields = [
        'asym_id_1',
        'seq_id_1',
        'asym_id_2',
        'seq_id_2',
        'distance_threshold'
        ]
    xlink_rows = xlink_restraint.fields(xlink_fields)
    xlinks = [((asym_id_1, int(seq_id_1)), (asym_id_2, int(seq_id_2)), float(distance_threshold))
              for asym_id_1, seq_id_1, asym_id_2, seq_id_2, distance_threshold in xlink_rows]
    if xlinks:
#        xname = '%s crosslinks' % name
#        g = session.pb_manager.get_group(xname)
#        g.name = xname
        xlink_radius = 1.0
        long_xlink_color = (255,0,0,255)	# Red
        xlink_color = (0,255,0,255)		# Green
        for m in models:
            xname = '%s crosslinks' % m.name
            g = m.pseudobond_group(xname)
            g.name = xname
            for r1, r2, d in xlinks:
                s1, s2 = m.residue_sphere(*r1), m.residue_sphere(*r2)
                if s1 and s2 and s1 is not s2:
                    b = g.new_pseudobond(s1, s2)
                    b.color = long_xlink_color if b.length > d else xlink_color
                    b.radius = xlink_radius
                    b.halfbond = False
                    b.restraint_distance = d
#        models.append(g)

    return models, ('Opened IHM file %s containing %d sphere models, %d distance restraints' %
                    (filename, len(mspheres), len(xlinks)))

# -----------------------------------------------------------------------------
#
def register():
    from chimerax.core import io
    from chimerax.core.atomic import structure
    io.register_format("Integrative Hybrid Model", structure.CATEGORY, (".ihm",), ("ihm",),
                       open_func=read_ihm)

# -----------------------------------------------------------------------------
#
from chimerax.core.atomic import Structure
class IHMSphereModel(Structure):
    def __init__(self, session, name, sphere_list):
        Structure.__init__(self, session, name = name, smart_initial_display = False)

        self._res_sphere = rs = {}	# (asym_id, res_num) -> sphere atom
        
        from chimerax.core.colors import chain_rgba8
        for (sb,se,asym_id,xyz,r) in sphere_list:
            aname = ''
            a = self.new_atom(aname, 'H')
            a.coord = xyz
            a.radius = r
            a.draw_mode = a.SPHERE_STYLE
            a.color = chain_rgba8(asym_id)
            rname = '%d' % (se-sb+1)
            r = self.new_residue(rname, asym_id, sb)
            r.add_atom(a)
            for s in range(sb, se+1):
                rs[(asym_id,s)] = a
        self.new_atoms()

    def residue_sphere(self, asym_id, res_num):

        return self._res_sphere.get((asym_id, res_num))
    
