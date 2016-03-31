# vim: set expandtab shiftwidth=4 softtabstop=4:

from ..models import Model
class Alignment(Model):
    def __init__(self, session, seqs, file_attrs=None, file_markups=None):
        Model.__init__(self, "alignment", session)
        self.seqs = seqs

    def child_models(self):
        return self.seqs

# parser_types is used if you need to read a file to get it's sequences without creating
# an alignment (e.g. align additional sequences into an existing alignment)
parser_types = {}

def _parse_to_models(session, file_name, parser):
    seqs, file_attrs, file_markups = parser(file_name)
    seq0_len = len(seqs[0])
    for seq in seqs[1:]:
        if len(seq) != seq0_len:
            aligned = False
            break
    else:
        aligned = True
    if aligned:
        return[Alignment(session, seqs, file_attrs=file_attrs, file_markups=file_markups)]
    return[Alignment([s]) for s in seqs]

def register_formats():
    registration_info = {}
    import os
    # sort so that we see the 'read_' modules before the 'write_' modules
    for f in sorted(list(os.listdir(os.path.join(os.path.dirname(__file__), "io")))):
        if f.startswith("read_") and f.endswith(".py"):
            format_name = f[5:-3]
            temp_ns = {}
            exec("from .io.read_{} import parse, extensions, description".format(format_name),
                globals(), temp_ns)
            kw = {'open_func': lambda ses, fn, *args, parser=temp_ns['parse']:
                _parse_to_models(ses, fn, parser)}
            extensions = temp_ns['extensions']
            description = temp_ns['description']
            parser_types[format_name] = (parser, extensions)
        elif f.startswith("write_") and f.endswith(".py"):
            # assumes that there is a corresponding 'read_' module that was examined first
            format_name = f[6:-3]
            temp_ns = {}
            exec("from .io.write_{} import save".format(format_name), globals(), temp_ns)
            description, extensions, kw = registration_info[format_name]
            kw['export_func'] = temp_ns['save']
        else:
            continue
        registration_info[format_name] = (description, extensions, kw)
    from ..io import register_format, SEQUENCE
    for description, extensions, kw in registration_info.values():
        register_format(description, SEQUENCE, extensions, **kw)
