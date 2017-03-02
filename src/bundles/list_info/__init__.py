# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI

class _InfoAPI(BundleAPI):

    @staticmethod
    def register_command(command_name):
        # 'register_command' is lazily called when the command is referenced
        from . import cmd
        from chimerax.core.commands import register
        register('info',
                 cmd.info_desc,
                 cmd.info)
        register("info bounds",
                 cmd.info_bounds_desc,
                 cmd.info_bounds)
        register("info models",
                 cmd.info_models_desc,
                 cmd.info_models)
        register("info chains",
                 cmd.info_chains_desc,
                 cmd.info_chains)
        register("info polymers",
                 cmd.info_polymers_desc,
                 cmd.info_polymers)
        register("info residues",
                 cmd.info_residues_desc,
                 cmd.info_residues)
        register("info atoms",
                 cmd.info_atoms_desc,
                 cmd.info_atoms)
        register("info selection",
                 cmd.info_selection_desc,
                 cmd.info_selection)
        register("info resattr",
                 cmd.info_resattr_desc,
                 cmd.info_resattr)
        register("info distmat",
                 cmd.info_distmat_desc,
                 cmd.info_distmat)
        register("info notify start",
                 cmd.info_notify_start_desc,
                 cmd.info_notify_start)
        register("info notify stop",
                 cmd.info_notify_stop_desc,
                 cmd.info_notify_stop)
        register("info notify suspend",
                 cmd.info_notify_suspend_desc,
                 cmd.info_notify_suspend)
        register("info notify resume",
                 cmd.info_notify_resume_desc,
                 cmd.info_notify_resume)

bundle_api = _InfoAPI()
