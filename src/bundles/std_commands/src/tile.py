#
# vi:set expandtab shiftwidth=4:
#

def tile(session, models=None, columns=None, spacing_factor=1.3, view_all=True):
    """Tile models onto a square(ish) grid."""

    models = [m for m in models if m.bounds() is not None]
    if len(models) == 0:
        from chimerax.core.errors import UserError
        raise UserError("No models found for tiling.")

    # Size each tile to the maximum size (+buffer) among all models
    spacing = max([m.bounds().radius() for m in models]) * spacing_factor

    # First model is the anchor
    anchor = models[0].bounds().center()
    anchor_spec = '#' + models[0].id_string

    # Move model offset from anchor model in screen coordinate system
    if columns is None:
        import math
        columns = int(round(math.sqrt(len(models))))

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
        from chimerax.core.commands import run
        run(session, "; ".join(commands), log=False)
    undo.finish(session, models)
    session.undo.register(undo)
    session.logger.info("%d model%s tiled" %
                        (len(models), "s" if len(models) != 1 else ""))


def register_command(logger):
    from chimerax.core.commands import CmdDesc, register
    from chimerax.core.commands import TopModelsArg, IntArg, BoolArg, FloatArg
    desc = CmdDesc(optional=[("models", TopModelsArg)],
                   keyword=[("columns", IntArg),
                            ("view_all", BoolArg),
                            ("spacing_factor", FloatArg)],
                   synopsis="tile models onto grid")
    register("tile", desc, tile, logger=logger)
