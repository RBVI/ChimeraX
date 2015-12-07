# vim: set expandtab shiftwidth=4 softtabstop=4:


def open(session, filename, format=None, name=None):
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
        Not implemented.  User-supplied name (as opposed to the filename).
    '''
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
    desc = CmdDesc(required=[('filename', StringArg)],
                   keyword=[('format', DynamicEnum(formats)),
                            ('name', StringArg),
                            #('id', ModelIdArg),
                        ],
                   synopsis='read and display data')
    register('open', desc, open)
