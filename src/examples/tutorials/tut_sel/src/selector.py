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
    # models is a list of chimerax.core.atomic.Model instances
    # results is an instance of chimerax.core.objects.Objects

    # Iterate through the models and add atoms that are end
    # residues in chains.  If model does not have chains, just
    # silently ignore it.
    for m in models:
        try:
            chains = m.chains
        except AttributeError:
            continue
        # chains is an instance of chimerax.core.atomic.molarray.Chains
        # whose elements are chimerax.core.atomic.molarray.Chain instances
        for c in chains:
            residues = c.residues
            # residues is an instance of chimerax.core.atomic.molarray.Residues
            # whose elements are chimerax.core.atomic.molarray.Residue instances
            # or None (if residue is listed in sequence but coordinate data
            # is missing).  To add the residues, we start from either
            # end and quit after the first "real"/non-None residue.
            # 'results' only holds models, atoms and bonds, not residues.
            # We add atoms from the residues on the ends.  Bonds between
            # atoms in each residue are also added.
            for r in residues:
                if r is not None:
                    results.add_atoms(r.atoms, bonds=True)
                    break
            for r in reversed(residues):
                if r is not None:
                    results.add_atoms(r.atoms, bonds=True)
                    break


# Map selector name to corresponding callback function.
# Only one selector for now, but more can be added
_selector_func = {
    "endres":   _select_endres
}
