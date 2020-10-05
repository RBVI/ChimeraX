#
# vi:set expandtab shiftwidth=4:
#

def tile(session, models=None, columns=None, spacing_factor=1.3, view_all=True):
    """Tile models onto a square(ish) grid."""

    # Keep only models that we think might work
    # Surfaces are excluded on the assumption that they will move
    # with their associated models
    from chimerax.atomic import Structure
    from chimerax.map.volume import Volume
    tilable_classes = [Structure, Volume]
    def tilable(m):
        for klass in tilable_classes:
            if isinstance(m, klass):
                return True
        return False
    if models is None:
        models = [m for m in session.models.list() if tilable(m)]
    models = [m for m in models if tilable(m) and m.bounds() is not None]
    if len(models) == 0:
        from chimerax.core.errors import UserError
        raise UserError("No models found for tiling.")

    # Size each tile to the maximum size (+buffer) among all models
    spacing = max([m.bounds().radius() for m in models]) * spacing_factor

    # First model is the anchor
    anchor = models[0].bounds().center()
    anchor_spec = '#' + models[0].id_string

    from .view import UndoView
    undo = UndoView("tile", session, models)
    with session.undo.block():
        # Simultaneously ove model toward anchor in scene coordinate system
        # and toward final target position in screen coordinate system
        import math, numpy
        from chimerax.geometry import norm
        if columns is None:
            columns = int(round(math.sqrt(len(models))))
        commands = []
        for i, m in enumerate(models[1:], start=1):
            center = m.bounds().center()
            # First move in scene coordinates to superimpose model onto anchor
            view_delta = anchor - center
            view_dist = norm(view_delta)
            if view_dist > 0:
                commands.append("move %.2f,%.2f,%.2f %.2f coord %s models %s" %
                                (view_delta[0], view_delta[1], view_delta[2],
                                 view_dist, anchor_spec, m.atomspec))
            # Then move in screen coordinates to final grid position
            row, col = divmod(i, columns)
            screen_delta = numpy.array([col * spacing, -row * spacing, 0],
                                       dtype=numpy.float32)
            screen_dist = norm(screen_delta)
            if screen_dist > 0:
                commands.append("move %.2f,%.2f,%.2f %.2f models %s" %
                                (screen_delta[0], screen_delta[1], screen_delta[2],
                                 screen_dist, m.atomspec))
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
    from chimerax.core.commands import ModelsArg, IntArg, BoolArg, FloatArg
    desc = CmdDesc(optional=[("models", ModelsArg)],
                   keyword=[("columns", IntArg),
                            ("view_all", BoolArg),
                            ("spacing_factor", FloatArg)],
                   synopsis="tile models onto grid")
    register("tile", desc, tile, logger=logger)
