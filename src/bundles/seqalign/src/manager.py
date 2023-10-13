# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

_builtin_subcommands = set(['chain', 'associate', 'disassociate'])
_viewer_subcommands = set()
_commands_registered = False

from chimerax.core.state import StateManager
from chimerax.core.toolshed import ProviderManager
class AlignmentsManager(StateManager, ProviderManager):
    """Manager for sequence alignments"""
    def __init__(self, session, name, bundle_info):
        self._alignments = {}
        # bundle_info needed for session save
        self.bundle_info = bundle_info
        self.session = session
        self.viewer_info = {'alignment': {}, 'sequence': {}}
        self.viewer_to_subcommand = {}
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger("new alignment")
        self.triggers.add_trigger("destroy alignment")
        self._headers = {}
        self._viewers = {}
        super().__init__(name)

    def __getitem__(self, i):
        '''index into models using square brackets (e.g. session.models[i])'''
        return list(self.alignments)[i]

    def __iter__(self):
        '''iterator over models'''
        return iter(self.alignments)

    def __len__(self):
        return len(self.alignments)

    def add_provider(self, bundle_info, name, *, type=None,
            synonyms=[], subcommand_name=None, sequence_viewer=True, alignment_viewer=True, **kw):
        """Register an alignment header, or an alignment/sequence viewer and its associated subcommand.

        Common Parameters
        ----------
        name : str
            Header or viewer name.
        type : str
            "header" or "viewer"

        Header Parameters
        ----------
        (none)

        'run_provider' for headers should return the header's class object.

        Viewer Parameters
        ----------
        sequence_viewer : bool
            Can this viewer show single sequences
        alignment_viewer : bool
            Can this viewer show sequence alignments
        synonyms : list of str
           Shorthands that the user could type instead of standard_name to refer to your tool
           in commands.  Example:  ['sv']
        subcommand_name : str
            If the viewer can be controlled by a subcommand of the 'sequence' command, the subcommand
            word to use after 'sequence'.

        'run_provider' for viewers should will receive an 'alignment' keyword argument with the alignment
            to view and should return the viewer instance.
        """
        if type == "header":
            self._headers[name] = bundle_info
        elif type == "viewer":
            if subcommand_name:
                if subcommand_name in _builtin_subcommands:
                    raise ValueError("Viewer subcommand '%s' is already a builtin"
                        " 'sequence' subcommand name" % subcommand_name)
                if subcommand_name in _viewer_subcommands:
                    raise ValueError("Viewer subcommand name '%s' is already taken" % subcommand_name)
                _viewer_subcommands.add(subcommand_name)
                self.viewer_to_subcommand[name] = subcommand_name
                if _commands_registered:
                    _register_viewer_subcommand(self.session.logger, subcommand_name)
            if synonyms:
                # comma-separated text -> list
                synonyms = [x.strip() for x in synonyms.split(',')]
            if sequence_viewer:
                self.viewer_info['sequence'][name] = synonyms
            if alignment_viewer:
                self.viewer_info['alignment'][name] = synonyms
            self._viewers[name] = bundle_info
        elif type is None:
            raise ValueError("Provider failed to specify type to alignments manager")
        else:
            raise ValueError("Alignments manager does not handle provider type '%s'" % type)

    @property
    def alignments(self):
        return list(self._alignments.values())

    @property
    def alignments_map(self):
        return {k:v for k,v in self._alignments.items()}

    def destroy_alignment(self, alignment):
        if alignment.ident is not False:
            del self._alignments[alignment.ident]
        self.triggers.activate_trigger("destroy alignment", alignment)
        alignment._destroy()

    def header(self, name):
        if name in self._headers:
            bundle_info = self._headers[name]
            if bundle_info.installed:
                return bundle_info.run_provider(self.session, name, self)
            raise ValueError("Alignment header '%s' is not installed" % name)
        raise ValueError("Unknown alignment header '%s'" % name)

    def headers(self):
        hdrs = []
        for name, info in self._headers.items():
            hdrs.append(self.header(name))
        return hdrs

    def header_names(self, *, installed_only=True):
        names = []
        for name, info in self._headers.items():
            bundle_info = info
            if installed_only and not bundle_info.installed:
                continue
            names.append(name)
        return names

    def new_alignment(self, seqs, identify_as, attrs=None, markups=None, auto_destroy=None,
            viewer=True, auto_associate=True, name=None, intrinsic=False, **kw):
        """Create new alignment from 'seqs'

        Parameters
        ----------
        seqs : list of :py:class:`~chimerax.atomic.Sequence` instances
            Contents of alignment
        identify_as : a text string (or None or False) used to identify the alignment in commands.
            If the string is already in use by another alignment, that alignment will be destroyed
            and replaced.  If identify_as is None, then a unique identifer will be generated and
            used.  The string cannot contain the ':' character, since that is used to indicate
            sequences within the alignment to commands.  Any such characters will be replaced
            with '/'.
            If False, then a "private" alignment will be returned that will not be shown in
            a viewer nor affected by any commands.
        auto_destroy : boolean or None
            Whether to automatically destroy the alignment when the last viewer for it
            is closed.  If None, then treated as False if the value of the 'viewer' keyword
            results in no viewer being launched, else True.
        viewer: str, True, or False
           What alignment/sequence viewer to launch.  If False, do not launch a viewer.  If True,
           use the current preference setting for the user.  The string must either be
           the viewer's tool display_name or a synonym registered by the viewer (during
           its register_viewer call).
        auto_associate : boolean or None
            Whether to automatically associate structures with the alignment.   A value of None
            is the same as False except that any StructureSeqs in the alignment will be associated
            with their structures.
        name : string or None
            Descriptive name of the alignment to use in viewer titles and so forth.  If not
            provided, same as identify_as.
        intrinsic : boolean
            If True, then the alignment is treated as "coupled" to the structures associated with
            it in that if all associations are removed then the alignment is destroyed.

        Returns the created Alignment
        """
        if self.session.ui.is_gui and identify_as is not False:
            viewer_text = viewer
            if len(seqs) > 1:
                attr = 'align_viewer'
                type_text = "alignment"
            else:
                attr = 'seq_viewer'
                type_text = "sequence"
            if viewer_text is True:
                from .settings import settings
                viewer_text = getattr(settings, attr).lower()
            if viewer_text:
                viewer_text = viewer_text.lower()
                for vname, syms in self.viewer_info[type_text].items():
                    if vname == viewer_text:
                        viewer_name = vname
                        break
                    if viewer_text in syms:
                        viewer_name = vname
                        break
                else:
                    self.session.logger.warning("No registered %s viewer corresponds to '%s'"
                        % (type_text, viewer_text))
                    viewer_text = False
        else:
            viewer_text = False
        if auto_destroy is None and viewer_text:
            auto_destroy = True

        from .alignment import Alignment
        if identify_as is None:
            i = 1
            while str(i) in self._alignments:
                i += 1
            identify_as = str(i)
        elif identify_as is not False and ':' in identify_as:
            self.session.logger.info(
                "Illegal ':' character in alignment identifier replaced with '/'")
            identify_as = identify_as.replace(':', '/')
        if identify_as in self._alignments:
            self.session.logger.info(
                "Destroying pre-existing alignment with identifier %s" % identify_as)
            self.destroy_alignment(self._alignments[identify_as])

        if name is None:
            from chimerax.atomic import StructureSeq
            if len(seqs) == 1 and isinstance(seqs[0], StructureSeq):
                sseq = seqs[0]
                if sseq.description:
                    description = "%s (%s)" % (sseq.description, sseq.full_name)
                else:
                    description = sseq.full_name
            else:
                description = identify_as
        elif identify_as is False:
            description = "private"
        else:
            description = name
        if identify_as:
            self.session.logger.info("Alignment identifier is %s" % identify_as)
        alignment = Alignment(self.session, seqs, identify_as, attrs, markups, auto_destroy,
            auto_associate, description, intrinsic, **kw)
        if identify_as:
            self._alignments[identify_as] = alignment
        if viewer_text:
            self._viewers[viewer_name].run_provider(self.session, viewer_name, self, alignment=alignment)
        self.triggers.activate_trigger("new alignment", alignment)
        return alignment

    @property
    def registered_viewers(self, seq_or_align):
        """Return the registered viewers of type 'seq_or_align'
            (which must be "sequence"  or "alignent")

           The return value is a list of tool names.
        """
        return list(self.viewer_info[seq_or_align].keys())

    def reset_state(self, session):
        for alignment in self._alignments.values():
            alignment._destroy()
        self._alignments.clear()

    @staticmethod
    def restore_snapshot(session, data):
        mgr = session.alignments
        mgr._ses_restore(data)
        return mgr

    def take_snapshot(self, session, flags):
        # viewer_info is "session independent"
        return {
            'version': 1,

            'alignments': self._alignments,
        }

    def _ses_restore(self, data):
        if self.session.restore_options['combine']:
            for ident in data.keys():
                try:
                    existing = self._alignments[ident]
                except KeyError:
                    continue
                existing._destroy()
            self._alignments.update(data['alignments'])
        else:
            self._alignments = data['alignments']

def _register_viewer_subcommands(logger):
    global _commands_registered
    _commands_registered = True
    for viewer_sub in _viewer_subcommands:
        _register_viewer_subcommand(logger, viewer_sub)

def _register_viewer_subcommand(logger, viewer_sub):
    def viewer_subcommand(session, alignment_s, subcommand_text, *, _viewer_keyword=viewer_sub):
        from .alignment import Alignment
        if alignment_s is None:
            from .cmd import get_alignment_by_id
            alignments = get_alignment_by_id(session, "", multiple_okay=True)
        elif isinstance(alignment_s, Alignment):
            alignments = [alignment_s]
        else:
            alignments = alignment_s
        for alignment in alignments:
            alignment._dispatch_viewer_command(_viewer_keyword, subcommand_text)
    from .cmd import AlignmentArg
    from chimerax.core.commands import CmdDesc, register, Or, EmptyArg, RestOfLine, ListOf
    desc = CmdDesc(
        required = [('alignment_s', Or(AlignmentArg,ListOf(AlignmentArg),EmptyArg)),
            ('subcommand_text', RestOfLine)],
        synopsis = "send subcommand to viewer '%s'" %viewer_sub
    )
    register('sequence %s' % viewer_sub, desc, viewer_subcommand, logger=logger)
