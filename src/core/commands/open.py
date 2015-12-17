# vim: set expandtab shiftwidth=4 softtabstop=4:


def open(session, filename, format=None, name=None, from_database=None, ignore_cache=False):
    '''Open a file.

    Parameters
    ----------
    filename : string
        A path to a file (relative to the current directory), or a database id code to
        fetch prefixed by the database name, for example, pdb:1a0m, mmcif:1jj2, emdb:1080.
        A 4-letter id that is not a local file is interpreted as an mmCIF fetch.
    format : string
        Read the file using this format, instead of using the file suffix to infer the format.
    name : string
        Name to use for data set.  If none specified then filename is used.
    from_database : string
        Database to fetch from. The filename is treated as a database identifier.
    ignore_cache : bool
        Whether to fetch files from cache.  Fetched files are always written to cache.
    '''

    if ':' in filename:
        prefix, fname = filename.split(':',maxsplit=1)
        from .. import fetch
        from_database, default_format = fetch.fetch_from_prefix(session, prefix)
        if from_database is None:
            from ..errors import UserError
            raise UserError('Unknown database prefix "%s" must be one of %s'
                            % (prefix, fetch.prefixes(session)))
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

    if from_database is not None:
        from .. import fetch
        if format is not None:
            db_formats = fetch.database_formats(session, from_database)
            if format not in db_formats:
                from ..errors import UserError
                raise UserError('Only formats %s can be fetched from database %s'
                                % (', '.join(db_formats), from_database))
        models = fetch.fetch_from_database(session, from_database, filename,
                                           format=format, name=name, ignore_cache=ignore_cache)
        if len(models) > 1:
            session.models.add_group(models)
        else:
            session.models.add(models)
        return models

    if format is not None:
        format = format_from_prefix(format)
    try:
        models = session.models.open(filename, format=format, name=name)
    except OSError as e:
        from ..errors import UserError
        raise UserError(e)

    return models

def format_from_prefix(prefix):
    from .. import io
    formats = [f for f in io.formats() if prefix in io.prefixes(f)]
    return formats[0]

def open_formats(session):
    '''Report file formats, suffixes and databases that the open command knows about.'''
    lines = ['<table border=1 cellspacing=0 cellpadding=2>', '<tr><th>File format<th>Suffixes']
    from .. import io
    formats = list(io.formats())
    formats.sort(key = lambda f: tuple(io.prefixes(f)))
    for f in formats:
        lines.append('<tr><td>%s<td>%s' % (' or '.join(io.prefixes(f)), ', '.join(io.extensions(f))))
    lines.append('</table>')
    lines.append('<p></p>')

    lines.extend(['<table border=1 cellspacing=0 cellpadding=2>', '<tr><th>Database<th>Formats'])
    from ..fetch import fetch_databases
    databases = list(fetch_databases(session).values())
    databases.sort(key = lambda k: k.database_name)
    for db in databases:
        formats = list(db.fetch_function.keys())
        formats.sort()
        formats.remove(db.default_format)
        formats.insert(0, db.default_format)
        line = '<tr><td>%s<td>%s' % (db.database_name, ', '.join(formats))
        pf = [(p,f) for p,f in db.prefix_format.items()
              if p != db.database_name]
        if pf:
            line += '<td>' + ', '.join('prefix %s fetches format %s' % (p,f) for p,f in pf)
        lines.append(line)
    lines.append('</table>')

    msg = '\n'.join(lines)
    session.logger.info(msg, is_html = True)

def register_command(session):
    from . import CmdDesc, register, DynamicEnum, StringArg, ModelIdArg, BoolArg
    def formats():
        from .. import io
        prefixes = sum((tuple(io.prefixes(f)) for f in io.formats()), ())
        return prefixes
    def db_formats():
        from .. import fetch
        return [f.database_name for f in fetch.fetch_databases(session).values()]
    desc = CmdDesc(required=[('filename', StringArg)],
                   keyword=[('format', DynamicEnum(formats)),
                            ('name', StringArg),
                            ('from_database', DynamicEnum(db_formats)),
                            ('ignore_cache', BoolArg),
                            #('id', ModelIdArg),
                        ],
                   synopsis='read and display data')
    register('open', desc, open)
    of_desc = CmdDesc(synopsis='report formats that can be opened')
    register('open formats', of_desc, open_formats)
