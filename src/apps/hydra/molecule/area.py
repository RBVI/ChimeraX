
def molecule_spheres(mlist = None, probe_radius = 1.4):
    atoms = molecule_atoms(mlist)
    centers = atoms.coordinates()
    radii = atoms.radii() + probe_radius
    return centers, radii

def molecule_atoms(mlist):
    from ..molecule import Atoms
    aset = Atoms()
    aset.add_molecules(mlist)
    aset = aset.exclude_water()
    return aset

def accessible_surface_area(molecule):
    centers, radii = molecule_spheres([molecule])
    from .. import surface
    area = surface.spheres_surface_area(centers, radii)
    return area

def area_command(cmdname, args, session):

    from ..commands.parse import atoms_arg, float_arg, parse_arguments
    req_args = (('atoms', atoms_arg),)
    opt_args = ()
    kw_args = (('probeRadius', float_arg),)

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    kw['session'] = session
    sasa(**kw)

def sasa(atoms, session, probeRadius = 1.4):
    centers = atoms.coordinates()
    radii = atoms.radii() + probeRadius
    from .. import surface
    areas = surface.spheres_surface_area(centers, radii)
    area = areas.sum()
    failed = (areas == -1).sum()
    if failed > 0:
        area = areas[areas >= 0].sum()
    msg = 'Solvent accessible area %.5g, for %d atoms' % (area, atoms.count())
    mols = list(atoms.molecules())
    mols.sort(key = lambda m: m.id)
    mids = '#' + ','.join('%d' % m.id for m in mols)
    msg += ', model%s %s' % (('s' if len(mols) > 1 else ''), mids)
    if failed > 0:
        msg += ', calculation failed for %d atoms' % failed
    session.show_status(msg)
    session.show_info(msg)

def test_sasa(n = 30, npoints = 1000):
#    test_pdb_models(pdbs, npoints)
#    return
#    test_all_pdb_models(['/Users/goddard/Downloads/Chimera/PDB'])
#    test_all_pdb_models(pdb_subdirs(), '.ent', npoints = npoints)
#    return

#    centers, radii = random_spheres_intersecting_unit_sphere(n)
#    from numpy import array
#    centers, radii = array(((0.0,0,0), (1.0,0,0))), array((1.0, 1.0))     # area test, pi
#    centers, radii = array(((0.0,0,0), (1.0,0,0), (0,1.0,0))), array((1.0, 1.0, 1.0)) # area test
#    centers, radii = array(((0.0,0,0), (1.0,0,0), (1.0,0,0))), array((1.0, 0.5, 0.25))  # Nested circle test
#    r = sqrt(2)
#    centers, radii = array(((0.0,0,0), (1.0,0,0))), array((1.0, r))     # area test, 2*pi
#    centers, radii = array(((0.0,0,0), (1.0,0,0), (0,1.0,0))), array((1.0, r, r))     # area test, 3*pi

#    buried_sphere_area(0, centers, radii, draw = True)

    centers, radii = molecule_spheres()

#    import cProfile
#    cProfile.runctx('print("area =", spheres_surface_area(centers, radii).sum())', globals(), locals())

#    i = 10
#    buried_sphere_area(i, centers, radii, draw = True)
    from .. import surface
    surface.spheres_surface_area(centers, radii)

# Example results.
# Testing on PDB 1a0m excluding waters, 242 atoms. 15 seconds for analytic area.  Most time culling circle intersections.
# Average of 36 circles per sphere, ~660 circle intersections per sphere, ~8 intersections on boundary.
# Error with 10000 sphere point estimate, max over each sphere 0.002, mean 0.0004 as fraction of full sphere area, time 5 seconds.

def pdb_subdirs(pdb_dir = '/usr/local/pdb'):
    from os import listdir
    from os.path import join
    subdirs = [join(pdb_dir,sd) for sd in listdir(pdb_dir) if len(sd) == 2]      # two letter subdirectories
    return subdirs

def test_all_pdb_models(pdb_dirs, pdb_suffix = '.pdb',
                        results = '/Users/goddard/ucsf/chimera2/src/hydra/sasa_results.txt',
                        npoints = 1000):

    from os import listdir
    from os.path import join
    from ..files import opensave
    from ..surface import surface_area_of_spheres, estimate_surface_area_of_spheres

    points, weights = sphere_points_and_weights(npoints)

    rf = open(results, 'a')
    rf.write('Area numerical estimate using %d sample points per sphere\n' % len(points))
    rf.write('%6s %6s %9s %9s %5s %5s %8s %8s\n' %
             ('PDB', 'atoms', 'area', 'earea', 'atime', 'etime', 'max err', 'mean err'))
    for pdb_dir in pdb_dirs:
        pdb_names = [p for p in listdir(pdb_dir) if p.endswith(pdb_suffix)]
        for p in pdb_names:
            opensave.open_files([join(pdb_dir,p)])
            centers, radii = molecule_spheres()
            from time import time
            t0 = time()
            areas = surface_area_of_spheres(centers, radii)
            t1 = time()
            eareas = estimate_surface_area_of_spheres(centers, radii, points, weights)
            t2 = time()
            nf = (areas == -1).sum()
            pdb_id = p[3:-4]
            if nf > 0:
                rf.write('%6s %6d %4d-fail %9.1f %5.2f %5.2f\n' %
                         (pdb_id, len(centers), nf, eareas.sum(), t1-t0, t2-t1))
            else:
                from numpy import absolute
                aerr = absolute(areas - eareas) / (4*pi*radii*radii)
                rf.write('%6s %6d %9.1f %9.1f %5.2f %5.2f %8.5f %8.5f\n' %
                         (pdb_id, len(centers), areas.sum(), eareas.sum(), t1-t0, t2-t1, aerr.max(), aerr.mean()))
            rf.flush()
            opensave.close_models()
    rf.close()

pdbs = ('3znu','2hq3','3zqy','2vb1','2yab','3ze1','3ze2','4baj','3ztp','1jxw','1jxu','3ziy','4bs0','4bpu','1jxt','3zcc','4bza','1jxy','1cod','2c04','1jxx','1utn','2xfg','4a2s','1cbn','2ynw','2wur','1alz','4bc5','1f5e','2i2h','2i2j','1olr','2xhn','2yoi','2izq','1hhu','2c03','4b5o','2x5n','2jc5','2ww7','2xjp','4b5v','145d','1hll','4hp2','3bxq','1bv8','1krw','4bf7','2fyl','2xy8','4i9y','1f3c','1j4o','1gbn','1hj8','2ypc','4alf','2yj8','2ynx','4ba7','2yiv','2j45','3ze6','2xu3','2v9l','3zuc','1s1h','2x7k','3zdj','2ynv','4av5','4bag','1jfp','1e3d','4ajx','4c4p','1mli','3fsp','3zbz','2glw','1iyw','2r6p','1tge','1mz0','1myz','3dll','2dfk','2wse','4axi','4l1p','1y1y','4e7u','2d86','2tci','2jfb','3tnq','3lrt',)


def test_pdb_models(id_codes, npoints, session):

    from ..files import opensave
    from ..surface import surface_area_of_spheres, estimate_surface_area_of_spheres

    points, weights = sphere_points_and_weights(npoints)

    print('Area numerical estimate using %d sample points per sphere\n' % len(points))
    print('%6s %6s %9s %9s %5s %5s %8s %8s\n' %
             ('PDB', 'atoms', 'area', 'earea', 'atime', 'etime', 'max err', 'mean err'))
    for id in id_codes:
            from ..molecule.fetch_pdb import fetch_pdb
            mlist = fetch_pdb(id)
            session.add_models(mlist)
            
            centers, radii = molecule_spheres()
            from time import time
            t0 = time()
            areas = surface_area_of_spheres(centers, radii)
            t1 = time()
            eareas = estimate_surface_area_of_spheres(centers, radii, points, weights)
            t2 = time()
            nf = (areas == -1).sum()
            if nf > 0:
                print('%6s %6d %4d-fail %9.1f %5.2f %5.2f\n' %
                         (id, len(centers), nf, eareas.sum(), t1-t0, t2-t1))
            else:
                from numpy import absolute
                aerr = absolute(areas - eareas) / (4*pi*radii*radii)
                print('%6s %6d %9.1f %9.1f %5.2f %5.2f %8.5f %8.5f\n' %
                         (id, len(centers), areas.sum(), eareas.sum(), t1-t0, t2-t1, aerr.max(), aerr.mean()))
            opensave.close_models()
