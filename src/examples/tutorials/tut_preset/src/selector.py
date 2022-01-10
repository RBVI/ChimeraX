# vim: set expandtab shiftwidth=4 softtabstop=4:

def register(selector_name, logger):
    """Register selector with ChimeraX.
    """
    # Registration is simply telling ChimeraX which function
    # to call when the selector is used.  If an unexpected
    # selector_name is given, the dictionary lookup will fail,
    # and the resulting exception will be caught by ChimeraX.
    from chimerax.core.commands import register_selector
    register_selector(selector_name, _selector_func[selector_name], logger)


def _select_endres(session, models, results):
    # session is an instance of chimerax.core.session.Session
    # models is a list of chimerax.atomic.Model instances
    # results is an instance of chimerax.core.objects.Objects

    # Iterate through the models and add atoms that are end
    # residues in chains.  If model does not have chains, just
    # silently ignore it.
    for m in models:
        try:
            chains = m.chains
        except AttributeError:
            continue
        # chains is an instance of chimerax.atomic.Chains
        # whose elements are chimerax.atomic.Chain instances
        for c in chains:
            residues = c.existing_residues
            # residues is an instance of chimerax.atomic.Residues
            # whose elements are chimerax.atomic.Residue
            # instances.
            # 'results' only holds models, atoms and bonds, not residues.
            # We add atoms from the residues on the ends.  Bonds between
            # atoms in each residue are also added.
            results.add_atoms(residues[0].atoms, bonds=True)
            results.add_atoms(residues[-1].atoms, bonds=True)


# Map selector name to corresponding callback function.
# Only one selector for now, but more can be added
_selector_func = {
    "endres":   _select_endres
}
