# vim: set expandtab shiftwidth=4 softtabstop=4:

def position(session, camera=None, models=None):
    '''
    Set model and camera positions. With no options positions are reported
    for the camera and all models. Positions are specified as 12-numbers,
    the rows of a 3-row, 4-column matrix where the first 3 columns are a
    rotation matrix and the last column is a translation applied after the rotation.

    Parameters
    ----------
    camera : Place
      Set the camera position.
    models : list of (Model, Place)
      Set model positions.
    '''
    v = session.main_view
    if camera is not None:
        v.camera.position = camera
    if models is not None:
        for m,p in models:
            m.position = p

    if camera is None and models is None:
        report_positions(session)

def position_initial(session, models=None):
    '''
    Set models to initial positions.

    Parameters
    ----------
    models : Models
      Set model positions to no rotation, no shift.
    '''

    if models is None:
        models = session.models.list()
    from ..geometry import Place
    for m in models:
        m.position = Place()

def report_positions(session):
    c = session.main_view.camera
    lines = ['camera position: %s' % _position_string(c.position)]
    mlist = session.models.list()
    if mlist:
        mpos = ','.join('#%s,%s' % (m.id_string(), _position_string(m.position)) for m in mlist)
        lines.append('model positions: %s\n' % mpos)
    session.logger.info('\n'.join(lines))

def _position_string(p):
    return ','.join('%.5g' % x for x in tuple(p.matrix.flat))

from . import Annotation
class ModelPlacesArg(Annotation):
    """Annotation for model id and positioning matrix as 12 floats."""
    name = "model positions"

    @staticmethod
    def parse(text, session):
        from . import cli
        token, text, rest = cli.next_token(text)
        fields = token.split(',')
        if len(fields) % 13:
            raise AnnotationError("Expected model id and 12 comma-separated numbers")
        mp = []
        while fields:
            tm, mtext, mrest = cli.TopModelsArg.parse(fields[0], session)
            if len(tm) == 0:
                raise AnnotationError('No models specified by "%s"' % fields[0])
            p = cli.PlaceArg.parse_place(fields[1:13])
            for m in tm:
                mp.append((m,p))
            fields = fields[13:]
        return mp, text, rest

def register_command(session):
    from . import CmdDesc, register, PlaceArg, ModelsArg, Or, NoArg
    desc = CmdDesc(
        keyword=[('camera', PlaceArg),
                 ('models', ModelPlacesArg)],
        synopsis='set camera and model positions')
    register('position', desc, position)
    desc = CmdDesc(
        optional=[('models', ModelsArg)],
        synopsis='set models to initial positions')
    register('position initial', desc, position_initial)
