# vim: set expandtab ts=4 sw=4:

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

class Viewer:
    """ Viewer displays a multiple sequence alignment """

    """
    buttons = ('Quit', 'Hide')
    help = "ContributedSoftware/multalignviewer/framemav.html"
    provideStatus = True
    statusWidth = 15
    statusPosition = "left"
    provideSecondaryStatus = True
    secondaryStatusPosition = "left"

    ConsAttr = "mavPercentConserved"

    MATCH_REG_NAME_START = "matches"
    ERROR_REG_NAME_START = "mismatches"
    GAP_REG_NAME_START = "missing structure"

    # so Model Loops tool can invoke it...
    MODEL_LOOPS_MENU_TEXT = "Modeller (loops/refinement)..."
    import RegionBrowser
    SEL_REGION_NAME = RegionBrowser.SEL_REGION_NAME

    def __init__(self, fileNameOrSeqs, fileType=None, autoAssociate=True,
                title=None, quitCB=None, frame=None, numberingDisplay=None,
                sessionSave=True):
    """
    def __init__(self, alignment):
        """ if 'autoAssocate' is None then it is the same as False except
            that any StructureSequences in the alignment will be associated
            with their structures
        """
        """
        from chimera import triggerSet
        self.triggers = triggerSet.TriggerSet()
        self.triggers.addTrigger(ADD_ASSOC)
        self.triggers.addTrigger(DEL_ASSOC)
        self.triggers.addTrigger(MOD_ASSOC)
        self.triggers.addHandler(ADD_ASSOC, self._fireModAssoc, None)
        self.triggers.addHandler(DEL_ASSOC, self._fireModAssoc, None)
        self.triggers.addTrigger(ADD_SEQS)
        self.triggers.addTrigger(PRE_DEL_SEQS)
        self.triggers.addTrigger(DEL_SEQS)
        self.triggers.addTrigger(ADDDEL_SEQS)
        self.triggers.addTrigger(SEQ_RENAMED)
        self.triggers.addHandler(ADD_SEQS, self._fireAddDelSeq, None)
        self.triggers.addHandler(DEL_SEQS, self._fireAddDelSeq, None)
        self.triggers.addHandler(ADDDEL_SEQS, self._fireModAlign, None)
        self.triggers.addTrigger(MOD_ALIGN)
        self.associations = {}
        self._resAttrs = {}
        self._edited = False

        from common import getStaticSeqs
        seqs, fileMarkups, fileAttrs = getStaticSeqs(fileNameOrSeqs, fileType=fileType)
        self.seqs = seqs
        self.prefs = prefs
        from SeqCanvas import shouldWrap
        if numberingDisplay:
            defaultNumbering = numberingDisplay
        else:
            defaultNumbering = (True,
                not shouldWrap(len(seqs), self.prefs))
        self.numberingsStripped = False
        if getattr(seqs[0], 'numberingStart', None) is None:
            # see if sequence names imply numbering...
            startInfo = []
            for seq in seqs:
                try:
                    name, numbering = seq.name.rsplit('/',1)
                except ValueError:
                    break
                try:
                    start, end = numbering.split('-')
                except ValueError:
                    start = numbering
                try:
                    startInfo.append((name, int(start)))
                except ValueError:
                    break
            if len(startInfo) == len(seqs):
                self.numberingsStripped = True
                for i, seq in enumerate(seqs):
                    seq.name, seq.numberingStart = \
                                startInfo[i]
            else:
                for seq in seqs:
                    if hasattr(seq, 'residues'):
                        for i, r in enumerate(seq.residues):
                            if r:
                                seq.numberingStart = r.id.position - 1
                                break
                        else:
                            seq.numberingStart = 1
                    else:
                        seq.numberingStart = 1
                if not numberingDisplay:
                    defaultNumbering = (False, False)

        self._seqRenameHandlers = {}
        for seq in seqs:
            self._seqRenameHandlers[seq] = seq.triggers.addHandler(
                seq.TRIG_RENAME, self._seqRenameCB, None)
        self._defaultNumbering = defaultNumbering
        self.fileAttrs = fileAttrs
        self.fileMarkups = fileMarkups
        if not title:
            if isinstance(fileNameOrSeqs, basestring):
                title = os.path.split(fileNameOrSeqs)[1]
            else:
                title = "MultAlignViewer"
        self.title = title
        self.autoAssociate = autoAssociate
        self.quitCB = quitCB
        self.sessionSave = sessionSave
        self._frame = frame
        self._runModellerWSList = []
        self._runModellerLocalList = []
        self._realignmentWSJobs = {'self': [], 'new': []}
        self._blastAnnotationServices = {}
        ModelessDialog.__init__(self)
        chimera.extension.manager.registerInstance(self)
        """
