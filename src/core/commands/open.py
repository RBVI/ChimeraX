# vim: set expandtab shiftwidth=4 softtabstop=4:

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

def open(session, filename, format=None, name=None, from_database=None, ignore_cache=False,
         smart_initial_display = True, trajectory = False, **kw):
    '''Open a file.

    Parameters
    ----------
    filename : string
        A path to a file (relative to the current directory), or a database id
        code to fetch prefixed by the database name, for example, pdb:1a0m,
        mmcif:1jj2, emdb:1080.  A 4-letter id that is not a local file is
        interpreted as an mmCIF fetch.
    format : string
        Read the file using this format, instead of using the file suffix to
        infer the format.
    name : string
        Name to use for data set.  If none specified then filename is used.
    from_database : string
        Database to fetch from. The filename is treated as a database
        identifier.
    ignore_cache : bool
        Whether to fetch files from cache.  Fetched files are always written
        to cache.
    smart_initial_display : bool
        Whether to display molecules with rich styles and colors.
    trajectory : bool
        Whether to read a PDB format multimodel file as coordinate sets (true)
        or as multiple models (false, default).
    '''

    if ':' in filename:
        prefix, fname = filename.split(':', maxsplit=1)
        from .. import fetch
        from_database, default_format = fetch.fetch_from_prefix(prefix)
        if from_database is not None:
            if format is None:
                format = default_format
            filename = fname
    elif from_database is None:
        # Accept 4 character filename without prefix as pdb id.
        from os.path import splitext
        base, ext = splitext(filename)
        if not ext and len(filename) == 4:
            from_database = 'pdb'
            if format is None:
                format = 'mmcif'

    kw.update({'smart_initial_display': smart_initial_display,
          'trajectory': trajectory})

    from ..filehistory import remember_file
    if from_database is not None:
        from .. import fetch
        if format is not None:
            db_formats = fetch.database_formats(from_database)
            if format not in db_formats:
                from ..errors import UserError
                from . import commas, plural_form
                raise UserError(
                    'Only %s %s can be fetched from %s database'
                    % (commas(['"%s"' % f for f in db_formats], ' and '),
                       plural_form(db_formats, "format"), from_database))
        models, status = fetch.fetch_from_database(session, from_database, filename,
                                                   format=format, name=name, ignore_cache=ignore_cache, **kw)
        if len(models) > 1:
            session.models.add_group(models)
        else:
            session.models.add(models)
        remember_file(session, filename, format, models, database=from_database)
        session.logger.status(status, log=True)
        if trajectory:
            report_trajectories(models, session.logger)
        return models

    if format is not None:
        fmt = format_from_name(format)
        if fmt:
            format = fmt.name

    from os.path import exists
    if exists(filename):
        paths = [filename]
    else:
        from glob import glob
        paths = glob(filename)
        if len(paths) == 0:
            from ..errors import UserError
            raise UserError('File not found: %s' % filename)

    try:
        models = session.models.open(paths, format=format, name=name, **kw)
    except OSError as e:
        from ..errors import UserError
        raise UserError(e)

    if trajectory:
        report_trajectories(models, session.logger)
    
    # Remember in file history
    remember_file(session, filename, format, models or 'all models')

    return models

def report_trajectories(models, log):
    for m in models:
        if hasattr(m, 'num_coord_sets'):
            log.info('%s has %d coordinate sets' % (m.name, m.num_coord_sets))

def format_from_name(name, open=True, save=False):
    from .. import io
    formats = [f for f in io.formats()
               if (name in f.nicknames or name == f.name) and
               ((open and f.open_func) or (save and f.export_func))]
    if formats:
        return formats[0]
    return None


def open_formats(session):
    '''Report file formats, suffixes and databases that the open command knows about.'''
    if session.ui.is_gui:
        lines = ['<table border=1 cellspacing=0 cellpadding=2>', '<tr><th>File format<th>Short name(s)<th>Suffixes']
    else:
        session.logger.info('File format, Short name(s), Suffixes:')
    from .. import io
    from . import commas
    formats = list(io.formats())
    formats.sort(key = lambda f: f.name)
    for f in formats:
        if session.ui.is_gui:
            lines.append('<tr><td>%s<td>%s<td>%s' % (f.name,
                commas(f.nicknames), ', '.join(f.extensions)))
        else:
            session.logger.info('    %s: %s: %s' % (f.name,
                commas(f.nicknames), ', '.join(f.extensions)))
    if session.ui.is_gui:
        lines.append('</table>')
        lines.append('<p></p>')

    if session.ui.is_gui:
        lines.extend(['<table border=1 cellspacing=0 cellpadding=2>', '<tr><th>Database<th>Formats'])
    else:
        session.logger.info('\nDatabase, Formats:')
    from ..fetch import fetch_databases
    databases = list(fetch_databases().values())
    databases.sort(key=lambda k: k.database_name)
    for db in databases:
        formats = list(db.fetch_function.keys())
        formats.sort()
        formats.remove(db.default_format)
        formats.insert(0, db.default_format)
        if not session.ui.is_gui:
            session.logger.info('    %s: %s' % (db.database_name, ', '.join(formats)))
            continue
        line = '<tr><td>%s<td>%s' % (db.database_name, ', '.join(formats))
        pf = [(p, f) for p, f in db.prefix_format.items()
              if p != db.database_name]
        if pf:
            line += '<td>' + ', '.join('prefix %s fetches format %s' % (p, f) for p, f in pf)
        lines.append(line)
    if session.ui.is_gui:
        lines.append('</table>')
        msg = '\n'.join(lines)
        session.logger.info(msg, is_html=True)


def register_command(session):
    from . import CmdDesc, register, DynamicEnum, StringArg, BoolArg, OpenFileNameArg, NoArg

    def formats():
        from .. import io
        names = sum((tuple(f.nicknames) for f in io.formats()), ())
        return names

    def db_formats():
        from .. import fetch
        return [f.database_name for f in fetch.fetch_databases().values()]
    desc = CmdDesc(
        required=[('filename', OpenFileNameArg)],
        keyword=[
            ('format', DynamicEnum(formats)),
            ('name', StringArg),
            ('from_database', DynamicEnum(db_formats)),
            ('ignore_cache', NoArg),
            ('smart_initial_display', BoolArg),
            ('trajectory', BoolArg),
            # ('id', ModelIdArg),
        ],
        synopsis='read and display data')
    register('open', desc, open)
    of_desc = CmdDesc(synopsis='report formats that can be opened')
    register('open formats', of_desc, open_formats)
