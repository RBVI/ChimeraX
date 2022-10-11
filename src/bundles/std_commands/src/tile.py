#
# vi:set expandtab shiftwidth=4:
#

def tile(session, models=None, columns=None, spacing_factor=1.3,
         view_all=True, independent_rotation=True):
    """Tile models onto a rectangular grid."""

    models = _tiling_models(session, models)

    # Size each tile to the maximum size (+buffer) among all models
    spacing = max([m.bounds().radius() for m in models]) * spacing_factor

    # First model is the anchor
    anchor = models[0].bounds().center()
    anchor_spec = '#' + models[0].id_string

    # Move model offset from anchor model in screen coordinate system
    if columns is None:
        import math
        columns = int(math.ceil(math.sqrt(len(models))))

    screen_to_scene = session.main_view.camera.position
    scene_to_screen = screen_to_scene.inverse()

    from .view import UndoView
    undo = UndoView("tile", session, models)
    with session.undo.block():
        commands = []
        for i, m in enumerate(models[1:], start=1):
            center = m.bounds().center()
            row, col = divmod(i, columns)
            offset = screen_to_scene.transform_vector([col * spacing, -row * spacing, 0])
            view_delta = scene_to_screen.transform_vector(anchor + offset - center)
            commands.append("move %.2f,%.2f,%.2f models %s" %
                            (view_delta[0], view_delta[1], view_delta[2], m.atomspec))
        # Make everything visible on screen
        if view_all:
            commands.append("view")
        if independent_rotation:
            commands.append('mouse left "rotate independent" ; light simple')
            session.triggers.add_handler('remove models', rotate_normal(models))
        from chimerax.core.commands import run
        run(session, "; ".join(commands), log=False)
    undo.finish(session, models)
    if independent_rotation:
        undo = UndoMouseMode(session, 'tile', 'left', 'rotate', 'rotate independent',
                             extra_undo = undo)
    session.undo.register(undo)

    session.logger.info("%d model%s tiled" %
                        (len(models), "s" if len(models) != 1 else ""))

class rotate_normal:
    '''
    Switch mouse mode from rotate independent to normal rotation after
    all tiled models are closed.
    '''
    def __init__(self, tiled_models):
        self._session = tiled_models[0].session if tiled_models else None
        self._tiled_models = set(tiled_models)
    def __call__(self, trigger_name, deleted_models):
        tiled = self._tiled_models
        for m in deleted_models:
            tiled.discard(m)
        if tiled:
            return  # Some tiled models still open

        s = self._session
        mode = s.ui.mouse_modes.mode(button = 'left')
        if mode and mode.name == 'rotate independent':
            from chimerax.core.commands import run
            run(s, 'mouse left rotate', log = False)

        return 'delete handler'
    
def _tiling_models(session, models):
    if models is None:
        models = [m for m in session.models.list() if len(m.id) == 1 and m.visible]

    if len(models) == 1:
        # If we have one grouping model then tile the child models.
        m = models[0]
        from chimerax.core.models import Model
        if m.empty_drawing() and type(m) is Model and len(m.child_models()) > 1:
            models = m.child_models()
            
    models = [m for m in models if m.bounds() is not None]
    if len(models) == 0:
        from chimerax.core.errors import UserError
        raise UserError("No models found for tiling.")

    return models

from chimerax.core.undo import UndoAction
class UndoMouseMode(UndoAction):
    def __init__(self, session, name, button, undo_mode, redo_mode, extra_undo = None):
        self._session = session
        self._button = button
        self._undo_mode = undo_mode
        self._redo_mode = redo_mode
        self._extra_undo = extra_undo
        UndoAction.__init__(self, name)

    def undo(self):
        from chimerax.core.commands import run
        run(self._session, 'mouse %s "%s"' % (self._button, self._undo_mode), log = False)
        if self._extra_undo:
            self._extra_undo.undo()

    def redo(self):
        from chimerax.core.commands import run
        run(self._session, 'mouse %s "%s"' % (self._button, self._redo_mode), log = False)
        if self._extra_undo:
            self._extra_undo.redo()

def untile(session, models=None, view_all=True):
    """Untile models."""
    models = _tiling_models(session, models)
    pos = models[0].scene_position
    for m in models[1:]:
        m.scene_position = pos
    if view_all:
        from chimerax.core.commands import run
        run(session, 'view', log=False)
    mm = session.ui.mouse_modes.mode('left')
    if mm and mm.name == 'rotate independent':
        from chimerax.core.commands import run
        run(session, 'mouse left rotate', log=False)


def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, create_alias
    from chimerax.core.commands import TopModelsArg, IntArg, BoolArg, FloatArg
    desc = CmdDesc(optional=[("models", TopModelsArg)],
                   keyword=[("columns", IntArg),
                            ("spacing_factor", FloatArg),
                            ("view_all", BoolArg),
                            ("independent_rotation", BoolArg)],
                   synopsis="tile models onto grid")
    register("tile", desc, tile, logger=logger)
    desc = CmdDesc(optional=[("models", TopModelsArg)],
                   keyword=[("view_all", BoolArg)],
                   synopsis="untile models")
    register('tile off', desc, untile, logger=logger)
    create_alias('~tile', 'tile off $*', logger=logger)
