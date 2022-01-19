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
        from chimerax.core.commands import run
        run(session, "; ".join(commands), log=False)
    undo.finish(session, models)
    session.undo.register(undo)

    session.logger.info("%d model%s tiled" %
                        (len(models), "s" if len(models) != 1 else ""))

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
