# vim: set expandtab shiftwidth=4 softtabstop=4:


def open(session, filename, format=None, name=None, from_database=None):
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
    '''

    if from_database is None and format is None:
        from os.path import splitext
        base, ext = splitext(filename)
        if not ext and len(filename) == 4:
            from_database = 'pdb'
            format = 'mmcif'

    if ':' in filename:
        prefix, fname = filename.split(':',maxsplit=1)
        from .. import fetch
        from_database, format = fetch.fetch_from_prefix(session, prefix)
        if from_database is None:
            raise UserError('Unknown database prefix "%s" must be one of %s'
                            % (prefix, fetch.prefixes(session)))
        filename = fname

    if from_database is not None:
        from .. import fetch
        models = fetch.fetch_from_database(session, from_database, filename,
                                           format=format, name=name)
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

def register_command(session):
    from . import CmdDesc, register, DynamicEnum, StringArg, ModelIdArg
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
                            #('id', ModelIdArg),
                        ],
                   synopsis='read and display data')
    register('open', desc, open)
