# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.toolshed import ProviderManager

class PlotValueError(ValueError):
    pass

class MDPlottingManager(ProviderManager):

    bools = ["true", "false"]
    exclude_info = {
        'solution': bools,
        'hydrogens': bools,
        'ligands': bools,
        # order here controls order in the popup menu
        'metals': ["true", "alkali", "false"],
    }

    def __init__(self, session):
        self.session = session
        self.providers = {}
        self._provider_bundles = {}
        self._ui_names = {}
        self._num_atoms = {}
        self._min_vals = {}
        self._max_vals = {}
        self._text_formats = {}
        self._excludes = {}
        self._need_ref_frames = {}
        super().__init__("MD plotting")

    def add_provider(self, bundle_info, name, *, ui_name=None, num_atoms=None, min_val=None, max_val=None,
            text_format="%g", exclude=None, need_ref_frame=None):
        # 'name' is the name used as an arg in the command
        # 'ui_name' is the name used in the tool interface (defaults to 'name')
        # 'num_atoms' indicates how many atoms are needed to compute the quantity (and therefore are
        #     needed in the command form).  If num_atoms is zero, any number of atoms is okay.  If
        #     num_atoms is omitted, then the quantity is a scalar (e.g. energy).
        # 'min_val'/'max_val' are suggested minimum / maximum values to use for the plotting axis;
        #     if omitted, the min/max of the data values (possibly enlarged to the next esthetic value)
        #     will be used.
        # 'text_format' is the formatting operator to convert the numeric plotting value to the text
        #     displayed in the corresponding table.  It can be "distance" or "angle" for values that
        #     are distances or angles, which will get a more specific treatment than a generic format.
        # 'exclude' is a list of types of atoms that can be optionally excluded by the user
        #     from consideration when computing the plotting values. Specified as a comma-separated
        #     string of "kind=default" entries in the Provider tag.  The default specifies what value
        #     the chooser widget should initally have for that kind of atom in the interface.  The
        #     possible kinds and their values are:
        #         solution (solvent and non-metal ions): true/false
        #         hydrogens: true/false
        #         ligands: true/false
        #         metals (metal ions): true/alkali/false
        if num_atoms is not None:
            try:
                num_atoms = int(num_atoms)
                assert num_atoms >= 0
            except (ValueError, AssertionError):
                raise ValueError("'num_atom' must omitted or be a non-negative integer.\n"
                    f"'num_atoms' for provider {name} is {num_atoms}")
        self._provider_bundles[name] = bundle_info
        self._ui_names[name] = name if ui_name is None else ui_name
        self._num_atoms[name] = num_atoms
        self._min_vals[name] = min_val if min_val is None else float(min_val)
        self._max_vals[name] = max_val if max_val is None else float(max_val)
        self._text_formats[name] = text_format
        self._excludes[name] = excludes = {}
        if exclude is not None:
            for kind_value in exclude.split(','):
                try:
                    kind, value = kind_value.split('=')
                except ValueError:
                    raise ValueError("'exclude' must be a comma-separated list of 'kind=default' entries.\n"
                        f"The entry {kind_value} has the wrong number of equal signs.")
                if kind not in self.exclude_info.keys():
                    raise ValueError("Unrecognized 'exclude' kind; supported kinds are: %s" %
                        ", ".join(list(self.exclude_info.keys())))
                if value not in self.exclude_info[kind]:
                    raise ValueError("Unrecognized 'exclude' value (%s) for kind '%s';"
                        " supported values are: %s" % (value, kind,
                        ", ".join(list(self.exclude_info[kind]))))
                excludes[kind] = value
        if need_ref_frame is None:
            self._need_ref_frames[name] = False
        else:
            if need_ref_frame not in self.bools:
                raise ValueError("Unrecognized 'need_ref_frame' value (%s) for provider %s;"
                    " must be '%s' or '%s'" % (need_ref_frame, name, *self.bools))
            self._need_ref_frames[name] = eval(need_ref_frame.capitalize())

    def excludes(self, provider_name):
        return self._excludes[provider_name]

    def get_values(self, provider_name, **kw):
        return self._provider_bundles[provider_name].run_provider(self.session,
            provider_name, self, **kw)

    def max_val(self, provider_name):
        return self._max_vals[provider_name]

    def min_val(self, provider_name):
        return self._min_vals[provider_name]

    def need_ref_frame(self, provider_name):
        return self._need_ref_frames[provider_name]

    def num_atoms(self, provider_name):
        return self._num_atoms[provider_name]

    @property
    def provider_names(self):
        return list(self._provider_bundles.keys())

    def text_format(self, provider_name):
        tf = self._text_formats[provider_name]
        if tf == "distance":
            return self.session.pb_dist_monitor.distance_format
        elif tf == "angle":
            return "%.1f\N{DEGREE SIGN}"
        return tf

    def ui_name(self, provider_name):
        return self._ui_names[provider_name]

_plotting_manager = None
def get_plotting_manager(session):
    global _plotting_manager
    if _plotting_manager is None:
        _plotting_manager = MDPlottingManager(session)
    return _plotting_manager
