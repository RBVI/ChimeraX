#
# vi:set expandtab shiftwidth=4:
#

def tile(session, models=None, columns=None, spacing_factor=1.3, view_all=True):
    """Tile models onto a square(ish) grid."""

    # Keep only non-container models
    if models is None:
        models = session.models.list()
    models = [m for m in models if len(m.child_models()) == 0]
    from chimerax.std_commands.view import NamedView
    view = session.main_view
    before = NamedView(view, view.center_of_rotation, models)

    # Size each tile to the maximum size (+buffer) among all models
    spacing = max([m.bounds().radius() for m in models]) * spacing_factor

    # First model is the anchor
    anchor = models[0].bounds().center()
    anchor_spec = models[0].atomspec

    # Simultaneously ove model toward anchor in scene coordinate system
    # and toward final target position in screen coordinate system
    import math, numpy
    from chimerax.core.geometry import norm
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

    after = NamedView(view, view.center_of_rotation, models)
    session.undo.register(UndoTile("tile", before, after, session))


from chimerax.core.undo import UndoAction
class UndoTile(UndoAction):

    def __init__(self, name, before, after, session):
        super().__init__(name)
        self._before = before
        self._after = after
        self._session = session

    def undo(self):
        from chimerax.std_commands.view import view
        view(self._session, objects=self._before, frames=10)

    def redo(self):
        from chimerax.std_commands.view import view
        view(self._session, objects=self._after, frames=10)



def register_command(logger):
    from chimerax.core.commands import CmdDesc, register
    from chimerax.core.commands import ModelsArg, IntArg, BoolArg, FloatArg
    desc = CmdDesc(optional=[("models", ModelsArg)],
                   keyword=[("columns", IntArg),
                            ("view_all", BoolArg),
                            ("spacing_factor", FloatArg)],
                   synopsis="tile models onto grid")
    register("tile", desc, tile, logger=logger)
