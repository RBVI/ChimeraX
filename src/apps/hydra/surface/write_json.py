# -----------------------------------------------------------------------------
#
def write_json_command(cmdname, args, session):

    from ..commands.parse import path_arg, models_arg, bool_arg, parse_arguments
    req_args = (('path', path_arg),)
    opt_args = (('models', models_arg),)
    kw_args = ()

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    kw['session'] = session
    write_surfaces_as_json(**kw)

def write_surfaces_as_json(path, session, models = None):
    surfs = []
    nt = 0
    if models is None:
        models = session.top_level_models()
    for m in models:
        for d, positions in displayed_drawings(m):
            if not d.vertices is None:
                for p in positions:
                    va = p.moved(d.vertices)
                    na = p.apply_without_translation(d.normals)
                    s = {'vertices': bytes_to_string(va),
                         'normals': bytes_to_string(na),
                         'triangles': bytes_to_string(d.triangles),
                         'color': tuple(c/255 for c in d.color),
                     }
                    surfs.append(s)
                    nt += len(d.triangles)
    
    from io import StringIO
    j = StringIO()
    import json
    json.dump(surfs, j)
    f = open(path,'w')
    f.write(j.getvalue())
    f.close()
    nb = len(j.getvalue())
    print('wrote', len(surfs), 'surfaces', nb, 'bytes', nt, 'triangles', nb/nt, 'bytes per triangle')

def displayed_drawings(d):
    if not d.display:
        return []
    dd = [(d,d.positions)]
    for cd in d.child_drawings():
      dd.extend([(d2,d.positions*cpositions) for d2, cpositions in displayed_drawings(cd)])
    return dd

def bytes_to_string(b):
    from base64 import b64encode
    return b64encode(b).decode('utf-8')
