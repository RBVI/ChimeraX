# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

species_names = {
        "yeast":"saccharomyces cerevisiae",
        "human":"homo sapiens",
        "mouse":"mus musculus",
        "frog":"xenopus laevis",
        "worm":"caenorhabditis elegans",
        "fly":"drosophila melanogaster",
        "zebrafish":"danio rerio"}

from chimerax.cytoscape.cmd import send_command
from . import get_species
def stringdb(session, structures=None, species=None, cutoff=0.4, partners=10):
    if structures is None:
        from chimerax.core.commands import atomspec
        structures = structures.everything(session)
    for structure in structures:
        if species == None:
              # atomspec = spec.evaluate(session)
              #for model in atomspec.models
              #    print("species for %s are %s"%(model, get_species.get_species(model)))
              species_dict = get_species.get_species(structure)
              species_max = {}
              species = None
              max_count = -1
              for chain in species_dict.keys():
               s = species_dict[chain]
               if s in species_max:
                   species_max[s] += 1
               else:
                   species_max[s] = 1
               if species_max[s] > max_count:
                   species = s
                   max_count = species_max[s]
        else:
            # Substitute for command names
            if species in species_names:
                species = species_names[species]
        query = structure.name
        response = send_command(session, "structureViz", "stopListening", None)
        args = {"cutoff":str(cutoff),"limit":str(partners),"species":species,"query":query}
        response = send_command(session, "string", "protein query", args)
        for line in response:
            session.logger.info(line,is_html=True)
        response = send_command(session, "structureViz", "startListening", None)


from chimerax.core.commands import CmdDesc, AtomicStructuresArg, StringArg, IntArg, FloatArg
stringdb_desc = CmdDesc(optional=[("structures", AtomicStructuresArg)],
                        keyword=[("species", StringArg),
                                 ("cutoff", FloatArg),
                                 ("partners", IntArg)],
                        synopsis="Load interaction partners from string-db")
