# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI

class _MyAPI(BundleAPI):

    @staticmethod
    def register_command(command_name, bundle_info):
        # 'register_command' is lazily called when the command is referenced
        from . import cmd
        from chimerax.core.commands import register
        register("listinfo notify",
                 cmd.listinfo_notify_desc,
                 cmd.listinfo_notify)
        register("listinfo models",
                 cmd.listinfo_models_desc,
                 cmd.listinfo_models)
        register("listinfo chains",
                 cmd.listinfo_chains_desc,
                 cmd.listinfo_chains)
        register("listinfo polymers",
                 cmd.listinfo_polymers_desc,
                 cmd.listinfo_polymers)
        register("listinfo residues",
                 cmd.listinfo_residues_desc,
                 cmd.listinfo_residues)
        register("listinfo atoms",
                 cmd.listinfo_atoms_desc,
                 cmd.listinfo_atoms)
        register("listinfo selection",
                 cmd.listinfo_selection_desc,
                 cmd.listinfo_selection)
        register("listinfo resattr",
                 cmd.listinfo_resattr_desc,
                 cmd.listinfo_resattr)
        register("listinfo distmat",
                 cmd.listinfo_distmat_desc,
                 cmd.listinfo_distmat)

bundle_api = _MyAPI()
