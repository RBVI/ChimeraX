class File_Reader:
    def __init__(self, name, suffixes, reader, batch = False):
        self.name = name
        self.suffixes = suffixes
        self.reader = reader
        self.batch = batch

def file_readers(session):
    '''
    Return a list of file readers, each reader being represented by 3-tuple
    consisting of a file type name, a list of recognized suffixes, and a function
    that opens that file type give a path.  The funtion returns a list of models
    which have not been added to the scene.
    '''
    ftypes = session.file_readers
    if ftypes is None:
        from ..molecule.pdb import open_pdb_file
        from ..molecule.mmcif import open_mmcif_file
        from ..surface.read_stl import read_stl
        from .read_apr import open_autopack_results, read_ingredient_file, read_sphere_file
        from .read_swc import read_swc
        from ..surface.collada import read_collada_surfaces

        r = File_Reader
        ftypes = [
            r('PDB', ['pdb','ent'], open_pdb_file),
            r('mmCIF', ['cif'], open_mmcif_file),
            r('Session', ['hy'], open_session),
            r('AutoPack', ['apr'], open_autopack_results),
            r('AutoPack Ingredient', ['xml'], read_ingredient_file),
            r('AutoPack sphere file', ['sph'], read_sphere_file),
            r('STL', ['stl'], read_stl),
            r('Collada', ['dae'], read_collada_surfaces),
            r('Neuron SWC', ['swc'], read_swc),
            r('Python', ['py'], read_python),
        ]
        # Add map file types
        from ..map.data.fileformats import file_types as mft
        map_readers = [r(d, suffixes, open_map, batch = batch) for d,t,prefixes,suffixes,batch in mft]
        ftypes.extend(map_readers)
        session.file_readers = ftypes
    return ftypes

def file_reader_table(session):
    t = {}
    for r in file_readers(session):
        for s in r.suffixes:
            t['.' + s] = r
    return t

def file_writers(session):
    from . import session_file
    from ..surface import write_stl, write_json
    from ..map.data import fileformats
    w = {'.png': save_image_command,
         '.jpg': save_image_command,
         '.ppm': save_image_command,
         '.bmp': save_image_command,
         '.hy': lambda cmdname,path,session: session_file.save_session(path, session),
         '.mrc': fileformats.save_map_command,
         '.stl': write_stl.write_stl_command,
         '.json': write_json.write_json_command,
         }
    return w

def open_files(paths, session, set_camera = None):
    '''
    Open data files and add the models created to the scene.  The file types are recognized
    using the file suffix as listed in the list returned by file_readers().
    '''
    if set_camera is None:
        set_camera = (session.model_count() == 0)
    fr = file_reader_table(session)
    opened = []
    models = []
    batched_paths = set()
    from os.path import splitext, isfile
    for path in paths:
        ext = splitext(path)[1]
        if not isfile(path):
            session.show_status('File not found "%s"' % path)
            # TODO issue warning.
        elif ext in fr:
            r = fr[ext]
            if path in batched_paths:
                continue
            elif r.batch:
                bpaths = [p for p in paths if p.endswith(ext)]
                for p in bpaths:
                    batched_paths.add(p)
                mlist = r.reader(bpaths, session)
            else:
                mlist = r.reader(path, session)
            if not isinstance(mlist, (list, tuple)):
                mlist = [mlist]
            models.extend(mlist)
            opened.append(path)
            if set_camera and r.reader == open_session:
                set_camera = False
        else:
            session.show_status('Unknown file suffix "%s"' % ext)
            # TODO issue warning.
    session.add_models(models)
    finished_opening(opened, set_camera, session)
    return models

def finished_opening(opened, set_camera, session):
    s = session
    if opened and set_camera:
        view = s.view
        view.remove_overlays()
        view.initial_camera_view()

def report_opening(opened, session):
    s = session
    if len(opened) == 1 and opened:
        msg = 'Opened %s' % opened[0]
        s.show_info(msg, color = '#000080')
        s.show_status(msg)
    elif len(opened) > 1:
        msg = 'Opened %d files' % len(opened)
        s.show_info(msg, color = '#000080')
        s.show_status(msg)

def open_map(map_path, session):
    '''
    Open a density map file having any of the known density map formats.
    '''
    maps = []
    from ..map import data, volume_from_grid_data
    grids = data.open_file(map_path)
    for i,d in enumerate(grids):
        show = (i == 0)
        v = volume_from_grid_data(d, session, open_model = False, show_data = show)
        v.new_region(ijk_step = (1,1,1), adjust_step = False, show = show)
        maps.append(v)
    if len(maps) > 1:
        from os.path import basename
        name = basename(map_path if isinstance(map_path, str) else map_path[0])
        from ..map.series import Map_Series
        ms = Map_Series(name, maps)
        from ..map.series import slider
        slider.show_slider_on_open(session)
        return [ms]
    return maps

def open_session(path, session):
    '''
    Open a session file.  The current session is closed.
    '''
    from . import session_file
    session_file.restore_session(path, session)
    session.last_session_path = path
    session.show_info('Opened %s' % path, color = '#000080')
    return []

def save_session(session):
    '''
    Save a session file using the session file path of the last loaded
    session, or if no session has been loaded then show a dialog to
    get the save path.
    '''
    if session.last_session_path is None:
        from .. import ui
        ui.save_session_dialog(session)
    else:
        from . import session_file
        session_file.save_session(session.last_session_path, session)

def save_command(cmdname, args, session):
    a0 = args.split()[0]
    from os.path import splitext
    e = splitext(a0)[1]
    fw = file_writers(session)
    if e in fw:
        save_func = fw[e]
        save_func(cmdname, args, session)
    else:
        session.show_status('Unknown save file type %s' % a0)

def save_image(path, session, width = None, height = None, format = None, supersample = None, log_info = True):
    '''
    Save an image of the current graphics window contents.
    '''
    view = session.view
    if path is None:
        from .. import ui
        path = ui.image_save_dialog(session)

    import os.path
    path = os.path.expanduser(path)         # Tilde expansion
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        from ..commands import parse
        raise parse.CommandError('Directory "%s" does not exist' % dir)

    if format is None:
        from os.path import splitext
        format = splitext(path)[1][1:].upper()
        if not format in ('PNG', 'JPG', 'PPM', 'BMP'):
            from ..commands import parse
            raise parse.CommandError('Unrecognized image file suffix "%s"' % format)
        if format == 'JPG':
            format = 'JPEG'

    i = view.image(width, height, supersample = supersample)
    i.save(path, format)
    if log_info:
        session.show_info('saved image %s' % path)

def save_image_command(cmdname, args, session):

    from ..commands.parse import string_arg, int_arg, parse_arguments
    req_args = (('path', string_arg),)
    opt_args = ()
    kw_args = (('width', int_arg),
               ('height', int_arg),
               ('supersample', int_arg),
               ('format', string_arg),)

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    kw['session'] = session
    save_image(**kw)

def open_command(cmdname, args, session):

    from ..commands.parse import string_arg, parse_arguments
    req_args = (('path', string_arg),)
    opt_args = ()
    kw_args = (('fromDatabase', string_arg),)

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    kw['session'] = session
    if 'fromDatabase' in kw:
        kw['from_database'] = kw['fromDatabase']
        kw.pop('fromDatabase')
    open_data(**kw)

def open_data(path, session, from_database = None, set_camera = None, history = True):

    db = from_database
    if not db is None:
        ids = path.split(',')
        mlist = open_from_database(ids, session, db, set_camera)
    else:
        from os.path import expanduser
        from glob import glob
        paths = sum([glob(expanduser(p)) for p in path.split(',')], [])
        p0 = paths[0] if paths else ''
        from os.path import isfile
        if isfile(p0):
            mlist = open_files(paths, session)
        else:
            ids = path.split(',')
            id0 = ids[0]
            if len(id0) != 4:
                session.show_status('Unknown file %s' % path)
                return []
            db = 'EMDB' if is_integer(id0) else 'PDB'
            mlist = open_from_database(ids, session, db, set_camera)

    session.main_window.show_graphics()
    if history and mlist:
        session.file_history.add_entry(path, from_database = db, models = mlist)
    return mlist

def open_from_database(ids, session, from_database, set_camera = None):

    if set_camera is None:
        set_camera = (session.model_count() == 0)
    from . import fetch
    mlist = []
#    from time import time
#    t0 = time()
    for id in ids:
        m = fetch.fetch_from_database(id, from_database, session)
        if isinstance(m, (list, tuple)):
            mlist.extend(m)
        else:
            mlist.append(m)
#    t1 = time()
#    print('opened in %.2f seconds, %s' % (t1-t0, ids))
    session.add_models(mlist)
    finished_opening([m.path for m in mlist], set_camera, session)
    return mlist

def close_command(cmdname, args, session):

    from ..commands.parse import models_arg, parse_arguments
    req_args = ()
    opt_args = (('models', models_arg),)
    kw_args = ()

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    session.close_models(**kw)

def read_python(path, session):
    '''
    Read a Python file and execute the code.
    '''
    f = open(path)
    code = f.read()
    f.close()
    ccode = compile(code, path, 'exec')
    globals = locals = {}
    exec(ccode, globals, locals)
    return []

def is_integer(s):
    try:
        int(s)
    except ValueError:
        return False
    return True
