# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.toolshed import BundleAPI

class _MutationScoresAPI(BundleAPI):

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is called by the toolshed on start up
        from . import ms_data
        ms_data.register_commands(logger)
        from . import ms_label
        ms_label.register_command(logger)
        from . import ms_define
        ms_define.register_command(logger)
        from . import ms_scatter_plot
        ms_scatter_plot.register_command(logger)
        from . import ms_stats
        ms_stats.register_command(logger)
        from . import ms_histogram
        ms_histogram.register_command(logger)
        from . import ms_umap
        ms_umap.register_command(logger)

    @staticmethod
    def run_provider(session, name, mgr):
        if mgr == session.open_command:
            if name == 'Mutation scores':
                from chimerax.open_command import OpenerInfo
                class MutationScoresInfo(OpenerInfo):
                    def open(self, session, path, file_name, **kw):
                        from .ms_csv_file import open_mutation_scores_csv
                        ms_data, message = open_mutation_scores_csv(session, path, **kw)
                        return [], message
                    @property
                    def open_args(self):
                        from chimerax.core.commands import BoolArg
                        from chimerax.atomic import UniqueChainsArg
                        return {'chains': UniqueChainsArg,
                                'allow_mismatches': BoolArg}
                return MutationScoresInfo()

            elif name == 'UniProt Variants':
                from chimerax.open_command import OpenerInfo
                class VariantScoresInfo(OpenerInfo):
                    def open(self, session, path, file_name, **kw):
                        from .uniprot_variants import open_uniprot_variant_scores
                        ms_data, message = open_uniprot_variant_scores(session, path, **kw)
                        return [], message
                    @property
                    def open_args(self):
                        from chimerax.core.commands import StringArg, BoolArg
                        from chimerax.atomic import UniqueChainsArg
                        return {'chains': UniqueChainsArg,
                                'allow_mismatches': BoolArg,
                                'identifier': StringArg}
                return MutationScoresInfo()

            elif name == 'uniprot_variants':
                from chimerax.open_command import FetcherInfo
                class UniProtVariantsInfo(FetcherInfo):
                    def fetch(self, session, uniprot_id, format_name, ignore_cache, **kw):
                        from . import uniprot_variants
                        mset, msg = uniprot_variants.fetch_uniprot_variants(session, uniprot_id, ignore_cache = ignore_cache, **kw)
                        if session.ui.is_gui and len(mset.score_names()) >= 2:
                            x_score_name, y_score_name = mset.score_names()[:2]
                            from .ms_scatter_plot import mutation_scores_scatter_plot
                            mutation_scores_scatter_plot(session, x_score_name, y_score_name, mset.name,
                                                         color_synonymous = False, bounds = False, replace = False)
                        return [], msg
                    @property
                    def fetch_args(self):
                        from chimerax.core.commands import StringArg, BoolArg
                        from chimerax.atomic import UniqueChainsArg
                        return {'chains': UniqueChainsArg,
                                'allow_mismatches': BoolArg,
                                'identifier': StringArg}
                return UniProtVariantsInfo()

            elif name == 'AlphaMissense':
                from chimerax.open_command import OpenerInfo
                class AlphaMissenseScoresInfo(OpenerInfo):
                    def open(self, session, path, file_name, **kw):
                        from .alpha_missense import open_alpha_missense_scores
                        ms_data, message = open_alpha_missense_scores(session, path, **kw)
                        return [], message
                    @property
                    def open_args(self):
                        from chimerax.core.commands import StringArg, BoolArg
                        from chimerax.atomic import UniqueChainsArg
                        return {'chains': UniqueChainsArg,
                                'allow_mismatches': BoolArg,
                                'identifier': StringArg}
                return AlphaMissenseScoresInfo()

            elif name == 'alpha_missense':
                from chimerax.open_command import FetcherInfo
                class AlphaMissenseInfo(FetcherInfo):
                    def fetch(self, session, uniprot_id, format_name, ignore_cache, **kw):
                        from .alpha_missense import fetch_alpha_missense_scores
                        mset, msg = fetch_alpha_missense_scores(session, uniprot_id, ignore_cache = ignore_cache, **kw)
                        if session.ui.is_gui:
                            from .ms_histogram import mutation_scores_histogram
                            mutation_scores_histogram(session, 'amiss', mset.name, scale = 'linear', bins = 50,
                                                      curve = False, synonymous = False, bounds = False, replace = False)
                        return [], msg
                    @property
                    def fetch_args(self):
                        from chimerax.core.commands import StringArg, BoolArg
                        from chimerax.atomic import UniqueChainsArg
                        return {'chains': UniqueChainsArg,
                                'allow_mismatches': BoolArg,
                                'identifier': StringArg}
                return AlphaMissenseInfo()

    # Map class name to class for session restore
    @staticmethod
    def get_class(class_name):
        if class_name == 'MutationSet':
            from .ms_data import MutationSet
            return MutationSet
        elif class_name == 'MutationScores':
            from .ms_data import MutationScores
            return MutationScores
        elif class_name == 'ScoreValues':
            from .ms_data import ScoreValues
            return ScoreValues
        elif class_name == 'MutationScoresManager':
            from .ms_data import MutationScoresManager
            return MutationScoresManager
        elif class_name == 'MutationScatterPlot':
            from .ms_scatter_plot import MutationScatterPlot
            return MutationScatterPlot
        elif class_name == 'MutationHistogram':
            from .ms_histogram import MutationHistogram
            return MutationHistogram
        elif class_name == 'MutationLabelSessionSave':
            from .ms_label import MutationLabelSessionSave
            return MutationLabelSessionSave

bundle_api = _MutationScoresAPI()
