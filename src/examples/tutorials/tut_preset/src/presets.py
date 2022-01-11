# vim: set expandtab shiftwidth=4 softtabstop=4:

def run_preset(session, name, mgr):
    """Run requested preset.
    """
    # Registration is simply telling ChimeraX which function
    # to call when the selector is used.  If an unexpected
    # selector_name is given, the dictionary lookup will fail,
    # and the resulting exception will be caught by ChimeraX.
    cmds = [ "size stickRadius 0.07 ballScale 0.18" ]
    from chimerax.atomic import all_atomic_structures
    structures = all_atomic_structures(session)
    if name == "thin sticks":
        cmds.append("style stick")
    elif name == "ball and stick":
        cmds.append("style ball")
    else:
        raise ValueError("No preset named '%s'" % name)
    mgr.execute('; '.join(cmds))

