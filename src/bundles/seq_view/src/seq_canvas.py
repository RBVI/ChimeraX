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

from .settings import SINGLE_PREFIX, ALIGNMENT_PREFIX
from chimerax.atomic import Sequence

"""TODO
from Consensus import Consensus
from Conservation import Conservation
import string
from prefs import WRAP_IF, WRAP_THRESHOLD, WRAP, LINE_WIDTH, BLOCK_SPACE
from prefs import FONT_NAME, FONT_SIZE, BOLD_ALIGNMENT
from prefs import LINE_SEP, COLUMN_SEP, TEN_RES_GAP
from prefs import CONSERVATION_STYLE, CSV_CLUSTAL_CHARS
from prefs import CONSENSUS_STYLE
from prefs import RC_CLUSTALX, RC_BLACK, RC_RIBBON, RC_CUSTOM_SCHEMES
from prefs import RESIDUE_COLORING, nonFileResidueColorings
from prefs import SEQ_NAME_ELLIPSIS
from prefs import STARTUP_HEADERS
from prefs import SHOW_RULER_AT_STARTUP
from chimera import replyobj
from chimera.misc import chimeraLabel
import tkFont
import Tkinter
import Pmw
import chimera

ADD_HEADERS = "add headers"
DEL_HEADERS = "delete headers"
SHOW_HEADERS = "show headers"
HIDE_HEADERS = "hide headers"
DISPLAY_TREE = "hide/show tree"
"""

class SeqCanvas:
    """'public' methods are only public to the SequenceViewer class.
       Access to SeqCanvas functions is made through methods of the
       SequenceViewer class.
    """

    """TODO
    EditUpdateDelay = 7000
    viewMargin = 2
    """
    def __init__(self, parent, sv, alignment):
        from Qt.QtWidgets import QGraphicsView, QGraphicsScene, QHBoxLayout, QShortcut
        from Qt.QtCore import Qt
        self.label_scene = QGraphicsScene()
        """
        self.label_scene.setBackgroundBrush(Qt.lightGray)
        """
        self.label_scene.setBackgroundBrush(Qt.white)
        self.label_view = QGraphicsView(self.label_scene)
        self.label_view.setAttribute(Qt.WA_AlwaysShowToolTips)
        self.main_scene = QGraphicsScene()
        self.main_scene.setBackgroundBrush(Qt.white)
        """if gray background desired...
        ms_brush = self.main_scene.backgroundBrush()
        from Qt.QtGui import QColor
        ms_color = QColor(240, 240, 240) # lighter gray than "lightGray"
        ms_brush.setColor(ms_color)
        self.main_scene.setBackgroundBrush(ms_color)
        """
        class CustomView(QGraphicsView):
            def __init__(self, scene, resize_cb=self.viewport_resized):
                self.__resize_cb = resize_cb
                QGraphicsView.__init__(self, scene)

            def resizeEvent(self, event):
                super().resizeEvent(event)
                self.__resize_cb()
        self.main_view = CustomView(self.main_scene)
        self.main_view.setAttribute(Qt.WA_AlwaysShowToolTips)
        #self.main_view.setMouseTracking(True)
        main_vsb = self.main_view.verticalScrollBar()
        label_vsb = self.label_view.verticalScrollBar()
        main_vsb.valueChanged.connect(label_vsb.setValue)
        label_vsb.valueChanged.connect(main_vsb.setValue)
        from Qt.QtGui import QKeySequence
        self._copy_shortcut = QShortcut(QKeySequence.StandardKey.Copy, parent)
        import sys
        self._copy_shortcut.activated.connect(lambda *args: self.sv.show_copy_sequence_dialog())
        """TODO
        self.labelCanvas = Tkinter.Canvas(parent, bg="#E4E4E4")
        self._vdivider = Tkinter.Frame(parent, bd=2, relief='raised')
        self._hdivider = Tkinter.Frame(parent, bd=0, relief='flat',
            background="black")
        # force dividers to show...
        Tkinter.Frame(self._vdivider).pack()
        Tkinter.Frame(self._hdivider, background='black').pack()
        self.mainCanvas = Tkinter.Canvas(parent, bg="#E4E4E4")

        self._vscrollMapped = self._hscrollMapped = 0
        self.horizScroll = Tkinter.Scrollbar(parent,
                            orient="horizontal")
        self.mainCanvas["xscrollcommand"] = self.horizScroll.set
        self.horizScroll["command"] = self.mainCanvas.xview

        self.vertScroll = Tkinter.Scrollbar(parent, orient="vertical")
        # hooking up the label canvas yscrollcommand does weird things
        # if it's not managed, so just the main canvas...
        self.mainCanvas["yscrollcommand"] = self.vertScroll.set
        self.vertScroll["command"] = self._multiScroll

        # scroll wheel...
        self.mainCanvas.bind('<MouseWheel>', lambda e:
                    self.vertScroll.event_generate('<MouseWheel>',
                    state=e.state, delta=e.delta))
        # X doesn't deliver MouseWheel events, instead uses Button 4/5
        # events.  Greg's workaround code in tkgui to translate them into
        # MouseWheel events doesn't deliver the events to widgets that
        # don't accept the focus, so I need to bind them explicitly.
        self.mainCanvas.bind('<4>',
            lambda event: self._multiScroll('scroll', -1, 'units'))
        self.mainCanvas.bind('<5>',
            lambda event: self._multiScroll('scroll', 1, 'units'))

        self.mainCanvas.bind('<Configure>', self._configureCB)

        """
        self.sv = sv
        self.alignment = alignment
        """TODO
        for trig in [ADD_HEADERS, DEL_HEADERS,
                SHOW_HEADERS, HIDE_HEADERS, DISPLAY_TREE]:
            self.sv.triggers.addTrigger(trig)
        parent.winfo_toplevel().configure(takefocus=1)
        parent.winfo_toplevel().focus()
        self.mainCanvas.configure(takefocus=1)
        parent.winfo_toplevel().bind('<Next>', self._pageDownCB)
        parent.winfo_toplevel().bind('<space>', self._pageDownCB)
        parent.winfo_toplevel().bind('<Prior>', self._pageUpCB)
        parent.winfo_toplevel().bind('<Shift-space>', self._pageUpCB)
        parent.winfo_toplevel().bind('<Left>', self._arrowCB)
        parent.winfo_toplevel().bind('<Right>', self._arrowCB)
        parent.winfo_toplevel().bind('<Up>', self._arrowCB)
        parent.winfo_toplevel().bind('<Down>', self._arrowCB)
        parent.winfo_toplevel().bind('<Escape>', self._escapeCB)
        parent.winfo_toplevel().bind('<<Copy>>', self._copyCB)
        """
        """
        self.font = tkFont.Font(parent,
            (self.sv.prefs[FONT_NAME], self.sv.prefs[FONT_SIZE]))
        """
        from Qt.QtGui import QFont, QFontMetrics
        self.font = QFont("Helvetica")
        self.emphasis_font = QFont(self.font)
        self.emphasis_font.setBold(True)
        self.font_metrics = QFontMetrics(self.font)
        self.emphasis_font_metrics = QFontMetrics(self.emphasis_font)
        # On Windows the maxWidth() of Helvetica is 39(!), whereas the width of 'W' is 14.
        # So, I have no idea what that 39-wide character is, but I don't care -- just use
        # the width of 'W' as the maximum width instead.
        font_width, font_height = self.font_metrics.horizontalAdvance('W'), self.font_metrics.height()
        self.label_view.setMinimumHeight(font_height)
        self.main_view.setMinimumHeight(font_height)
        # pad font a little...
        self.font_pixels = (font_width + 1, font_height + 1)
        self.show_numberings = [len(self.alignment.seqs) == 1, False]
        from Qt.QtCore import QTimer
        self._resize_timer = QTimer()
        self._resize_timer.timeout.connect(self._actually_resize)
        self._resize_timer.start(200)
        self._resize_timer.stop()
        """TODO
        self.treeBalloon = Pmw.Balloon(parent)
        self.tree = self._treeCallback = None
        self.treeShown = self.nodesShown = False
        self._residueHandlers = None
        """
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        layout.addWidget(self.label_view)
        #layout.addWidget(self._vdivider)
        layout.addWidget(self.main_view, stretch=1)
        parent.setLayout(layout)
        self.label_view.hide()
        #self._vdivider.hide()
        self.main_view.show()
        self._initial_layout = True
        self.layout_alignment()
        """TODO
        self.mainCanvas.grid(row=1, column=2, sticky='nsew')
        parent.columnconfigure(2, weight=1)
        parent.rowconfigure(1, weight=1)

        # make the main canvas a reasonable size
        left, top, right, bottom = map(int,
                self.mainCanvas.cget("scrollregion").split())
        totalWidth = right - left + 1
        if self.wrap_okay():
            self.mainCanvas.config(width=totalWidth)
        else:
            seven = self.mainCanvas.winfo_pixels("7i")
            self.mainCanvas.config(width=min(seven, totalWidth))
        totalHeight = bottom - top + 1
        three = self.mainCanvas.winfo_pixels("3i")
        height = min(three, totalHeight)
        self.mainCanvas.config(height=height)
        self.labelCanvas.config(height=height)

        # need to update label outline box colors when molecules change
        self._trigID = chimera.triggers.addHandler(
                    'Molecule', self._molChange, None)

        # for tracking delayed update of headers/attrs during editing
        self._editBounds = self._delayedAttrsHandler = None

        # for undo/redo
        self._checkPoint(fromScratch=True)

        from MAViewer import ADDDEL_SEQS, SEQ_RENAMED
        self._addDelSeqsHandler = self.sv.triggers.addHandler(
                ADDDEL_SEQS, self._addDelSeqsCB, None)
        self._seqRenamedHandler = self.sv.triggers.addHandler(
                SEQ_RENAMED, lambda *args: self._reformat(), None)
        """
        from chimerax.atomic import get_triggers
        self._handlers = [get_triggers().add_handler('changes', self._changes_cb)]

    """TODO
    def activeNode(self):
        return self.lead_block.treeNodeMap['active']
    """

    def _actually_resize(self):
        self._resize_timer.stop()
        self._reformat()
        self._update_scene_rects()


    """TODO
    def _addDelSeqsCB(self, trigName, myData, trigData):
        self._clustalXcache = {}
        for seq in self.alignment.seqs:
            try:
                cf = seq.color_func
            except AttributeError:
                continue
            break
        for seq in self.alignment.seqs:
            seq.color_func = cf
            if cf != self._cfBlack:
                self.recolor(seq)
        
    def addSeqs(self, seqs):
        #TODO: need to see if adding sequences changes wrap_okay;
        # if it doesn't, does it change line_width (due to numberings
        # possibly getting wider).  If either, then just reformat.
        # If not, then need to pass new numbering_widths through to
        # SeqBlock.add_seqs
        for seq in seqs:
            self.labelBindings[seq] = {
                '<Enter>': lambda e, s=seq:
                    self.sv.status(self.seqInfoText(s)),
                '<Double-Button>': lambda e, s=seq: self.sv._editSeqName(s)
            }
        self.sv.region_browser._preAddLines(seqs)
        self.lead_block.addSeqs(seqs)
        self.sv.region_browser.redraw_regions()

    def adjustScrolling(self):
        self._resizescrollregion()
        self._recomputeScrollers()
        
    def _arrowCB(self, event):
        if event.state & 4 != 4:
            if event.keysym == "Up":
                self.sv.region_browser.raiseRegion(
                        self.sv.currentRegion())
            elif event.keysym == "Down":
                self.sv.region_browser.lowerRegion(
                        self.sv.currentRegion())
            else:
                self.sv.status(
                    "Use control-arrow to edit alignment\n")
            return

        if event.keysym == "Up":
            self._undoRedo(False)
            return
        if event.keysym == "Down":
            self._undoRedo(True)
            return

        region = self.sv.currentRegion()
        if not region:
            replyobj.error("No active region.\n")
            return

        if region.name == "ChimeraX selection":
            replyobj.error(
                "Cannot edit using Chimera selection region\n")
            return

        if len(region.blocks) > 1:
            replyobj.error(
                "Cannot edit with multi-block region.\n")
            return

        line1, line2, pos1, pos2 = region.blocks[0]
        if line1 not in self.alignment.seqs:
            line1 = self.alignment.seqs[0]
        if line2 not in self.alignment.seqs:
            replyobj.error("Edit region does not contain any"
                        " editable sequences.\n")
            return

        if event.keysym == "Left":
            incr = -1
            start = pos1
            end = pos2 + 1
        else:
            incr = 1
            start = pos2
            end = pos1 - 1

        gapPos = start + incr
        seqs = self.alignment.seqs[self.alignment.seqs.index(line1)
                        :self.alignment.seqs.index(line2)+1]

        offset = 0
        if gapPos < 0 or gapPos >= len(line1):
            self.sv.status("Need to add columns to alignment to"
                " allow for requested motion.\nPlease wait...")
            # try to figure out the gap character
            # in use...
            gapChar = None
            for s in self.alignment.seqs:
                for c in str(s):
                    if not c.isalnum():
                        gapChar = c
                        break
                if gapChar is not None:
                    break
            else:
                gapChar = '.'
            num2add = 10
            if incr == -1:
                newSeqs = [gapChar * num2add + str(x)
                            for x in self.alignment.seqs]
                start += num2add
                end += num2add
                pos1 += num2add
                pos2 += num2add
                gapPos += num2add
                offset = num2add
            else:
                newSeqs = [str(x) + gapChar * num2add
                            for x in self.alignment.seqs]
            self.sv.realign(newSeqs, offset=offset,
                            markEdited=True)
            self.sv.status("Columns added")
        else:
            for seq in seqs:
                if seq[gapPos].isalnum():
                    replyobj.error("No all-gap column in"
                        " requested direction; cannot"
                        " move editing region.\n"
                        " Select a larger region to"
                        " continue motion.\n")
                    return

        motion = 0
        while True:
            motion += incr
            for seq in seqs:
                gapChar = seq[gapPos]
                for i in range(start, end, 0-incr):
                    seq[i+incr] = seq[i]
                seq[end+incr] = gapChar
            if event.state & 1 != 1: # shift key not down
                break
            nextGap = gapPos + incr
            if nextGap < 0 or nextGap >= len(line1):
                break
            for seq in seqs:
                if seq[nextGap].isalnum():
                    break
            else:
                start += incr
                end += incr
                gapPos = nextGap
                continue
            break
        self.sv._edited = True
        if incr == -1:
            left, right = gapPos, end-2-motion
        else:
            left, right = end+2-motion, gapPos
            
        self._editRefresh(seqs, left, right, region=region,
            lastBlock=[line1, line2, pos1+motion, pos2+motion])
        self._checkPoint(offset=offset, left=left, right=right)

    #TODO: also add them to settings Header category
    def addHeaders(self, headers):
        headers = [hd for hd in headers if hd not in self.headers]
        if not headers:
            return
        for hd in headers:
            self.labelBindings[hd] = {}
        self.headers.extend(headers)
        self.display_header.update({}.fromkeys(headers, False))
        self.sv.triggers.activateTrigger(ADD_HEADERS, headers)
        self.showHeaders(headers)

    def _associationsCB(self, trigName, myData, trigData):
        matchMaps = trigData[1]
        for mm in matchMaps:
            self.recolor(mm['aseq'])
    """

    def assoc_mod(self, aseq):
        '''alignment sequence has gained or lost associated structure'''
        self.lead_block.assoc_mod(aseq)

    def bbox_list(self, line1, line2, pos1, pos2, cover_gaps=True):
        '''return coords that bound given lines and positions'''
        return self.lead_block.bbox_list(line1, line2, pos1, pos2, cover_gaps)

    def bounded_by(self, x1, y1, x2, y2, *, exclude_headers=False):
        '''return lines and offsets bounded by given coords'''
        return self.lead_block.bounded_by(x1, y1, x2, y2, exclude_headers)

    """TODO
    def _attrsUpdateCB(self):
        self._delayedAttrsHandler = None
        self.sv.status("Updating residue attributes")
        self.sv.setResidueAttrs()
        self.sv.status("Residue attributes updated")

    def _checkPoint(self, fromScratch=False, checkChange=False, offset=0,
                        left=None, right=None):
        if fromScratch:
            self._checkPoints = []
            self._checkPointIndex = -1
        self._checkPoints = self._checkPoints[:self._checkPointIndex+1]
        chkpt = [s[:] for s in self.alignment.seqs]
        if checkChange:
            if chkpt == self._checkPoints[self._checkPointIndex][0]:
                return
        self._checkPoints.append(
            (chkpt, (offset, self.sv._edited, left, right)))
        self._checkPointIndex += 1

    def _configureCB(self, e):
        # size change; scrollbars?
        if hasattr(self, "_configureWait") and self._configureWait:
            self.mainCanvas.after_cancel(self._configureWait)
        # Windows/Mac can get into a configure loop somehow unless we
        # do an actual 'after' instead of after_idle
        self._configureWait = self.mainCanvas.after(100, lambda e=e:
            self._configureWaitCB(e))

    def _configureWaitCB(self, e):
        self._configureWait = None

        if e.width <= 1 or e.height <=1:
            # wait for a 'real' event
            return

        self._recomputeScrollers(e.width, e.height)

    """
    def _changes_cb(self, trig_name, changes):
        reasons = changes.residue_reasons()
        if 'number changed' in reasons or 'insertion_code changed' in reasons:
            modified = changes.modified_residues()
            structures = set(modified.unique_structures)
            for aseq in self.alignment.seqs:
                needs_update = False
                for chain in aseq.match_maps.keys():
                    if chain.structure in structures:
                        needs_update = True
                        break
                if needs_update:
                    starts = set([chain.numbering_start for chain in aseq.match_maps.keys()])
                    starts.discard(None)
                    if len(starts) == 1:
                        aseq.numbering_start = starts.pop()
                        self.refresh(aseq, update_attrs=False)

    @property
    def consensus_capitalize_threshold(self):
        return self.consensus.capitalize_threshold

    @consensus_capitalize_threshold.setter
    def consensus_capitalize_threshold(self, capitalize_threshold):
        self.consensus.capitalize_threshold = capitalize_threshold

    @property
    def consensus_ignores_gaps(self):
        return self.consensus.ignore_gaps

    @consensus_ignores_gaps.setter
    def consensus_ignores_gaps(self, ignore_gaps):
        self.consensus.ignore_gaps = ignore_gaps

    @property
    def conservation_style(self):
        return self.conservation.style

    @conservation_style.setter
    def conservation_style(self, style):
        self.conservation.style = style

    """TODO
    def _copyCB(self, e):
        region = self.sv.currentRegion()
        if region is None:
            copy = "\n".join([s.ungapped() for s in self.alignment.seqs])
        else:
            texts = {}
            for line1, line2, pos1, pos2 in region.blocks:
                try:
                    i1 = self.alignment.seqs.index(line1)
                except ValueError:
                    i1 = 0
                try:
                    i2 = self.alignment.seqs.index(line2)
                except ValueError:
                    continue
                for seq in self.alignment.seqs[i1:i2+1]:
                    text = "".join([seq[p] for p in range(pos1, pos2+1)
                                if seq.gapped2ungapped(p) is not None])
                    if text:
                        texts[seq] = texts.setdefault(seq, "") + text
            if not texts:
                self.sv.status("Active region is all gaps!", color="red")
                return
            copy = "\n".join([texts[seq] for seq in self.alignment.seqs
                            if seq in texts])
        self.mainCanvas.clipboard_clear()
        self.mainCanvas.clipboard_append(copy)
        if region is None:
            if len(self.alignment.seqs) > 1:
                self.sv.status("No current region; copied all sequences")
            else:
                self.sv.status("Sequence copied")
        else:
            self.sv.status("Region copied")

    def dehighlightName(self):
        self.lead_block.dehighlightName()

    #TODO: also remove them from settings Header category
    def deleteHeaders(self, headers):
        if not headers:
            return
        for header in headers:
            if header in self.alignment.seqs:
                raise ValueError(
                    "Cannot delete an alignment sequence")
            if header in self.builtinHeaders:
                raise ValueError("Cannot delete builtin header"
                            " sequence")
        self.hide_headers(headers)
        for hd in headers:
            del self.display_header[hd]
            self.headers.remove(hd)
        self.sv.triggers.activateTrigger(DEL_HEADERS, headers)
    """

    def destroy(self):
        self._resize_timer.stop()
        for handler in self._handlers:
            handler.remove()
        self._handlers.clear()
    """
        chimera.triggers.deleteHandler('Molecule', self._trigID)
        from MAViewer import ADDDEL_SEQS, SEQ_RENAMED
        self.sv.triggers.deleteHandler(ADDDEL_SEQS,
                        self._addDelSeqsHandler)
        self.sv.triggers.deleteHandler(SEQ_RENAMED,
                        self._seqRenamedHandler)
        if self._residueHandlers:
            chimera.triggers.deleteHandler('Residue',
                        self._residueHandlers[0])
            from MAViewer import MOD_ASSOC
            self.sv.triggers.deleteHandler(MOD_ASSOC,
                        self._residueHandlers[1])
        self.lead_block.destroy()
        
    def _editHdrCB(self):
        left, right = self._editBounds
        self._editBounds = None
        for header in self.headers:
            if not hasattr(header, 'alignChange'):
                continue
            if header.fastUpdate():
                # already updated
                continue
            self.sv.status("Updating %s header" % header.name)
            header.alignChange(left, right)
            self.refresh(header, left=left, right=right)
            self.sv.status("%s header updated" % header.name)

    def _editRefresh(self, seqs, left, right, region=None, lastBlock=None):
        for header in self.headers:
            if not hasattr(header, 'alignChange'):
                continue
            if not self.display_header[header]:
                continue
            if not header.fastUpdate():
                # header can't update quickly; delay it
                self.sv.status("Postponing update of %s header"
                            % header.name)
                if self._editBounds:
                    self._editBounds = (min(left,
                        self._editBounds[0]), max(
                        right, self._editBounds[1]))
                    self.mainCanvas.after_cancel(
                            self._editHdrHandler)
                else:
                    self._editBounds = (left, right)
                self._editHdrHandler = self.mainCanvas.after(
                    self.EditUpdateDelay, self._editHdrCB)
                continue
            header.alignChange(left, right)
            self.refresh(header, left=left, right=right,
                            updateAttrs=False)
        for seq in seqs:
            self.refresh(seq, left=left, right=right,
                            updateAttrs=False)
        if region:
            region.updateLastBlock(lastBlock)
        self.sv.region_browser.redraw_regions(just_gapping=True)
        if not self._editBounds:
            if self._delayedAttrsHandler:
                self.mainCanvas.after_cancel(
                        self._delayedAttrsHandler)
            self._delayedAttrsHandler = self.mainCanvas.after(
                self.EditUpdateDelay, self._attrsUpdateCB)
    def _escapeCB(self, event):
        if event.state & 4 != 4:
            self.sv.status(
                "Use control-escape to revert to unedited\n")
            return
        while self._checkPointIndex > 0:
            self._undoRedo(True)
    """

    def find_numbering_widths(self, line_width):
        lwidth = rwidth = 0
        if self.show_numberings[0]:
            base_num_blocks = int(len(self.alignment.seqs[0]) / line_width)
            blocks = base_num_blocks + (base_num_blocks != len(self.alignment.seqs[0]) / line_width)
            extent = (blocks - 1) * line_width
            for seq in self.lines:
                numbering_start = line_numbering_start(seq)
                if numbering_start is None:
                    continue
                offset = len([c for c in seq[:extent] if c.isalpha() or c == '?'])
                lwidth = max(lwidth, self.font_metrics.horizontalAdvance(
                    "%d " % (numbering_start + offset)))
            lwidth += 3
        if self.show_numberings[1]:
            for seq in self.lines:
                numbering_start = line_numbering_start(seq)
                if numbering_start is None:
                    continue
                offset = len(seq.ungapped())
                rwidth = max(rwidth, self.font_metrics.horizontalAdvance(
                    "  %d" % (numbering_start + offset)))
        return [lwidth, rwidth]

    """
    def headerDisplayOrder(self):
        return self.lead_block.lines[:-len(self.alignment.seqs)]
    """

    def hide_header(self, header):
        self.lead_block.hide_header(header)
        self.sv.region_browser.redraw_regions()
        self._update_scene_rects()
        
    def layout_alignment(self):
        """
        from chimerax.alignment_headers import registered_headers, DynamicStructureHeaderSequence
        for seq, defaultOn in registeredHeaders.values():
            header = seq(self.alignment)
            self.headers.append(header)
            if use_disp_default and defaultOn:
                startup_headers.add(header.name)
        if use_disp_default:
            self.sv.prefs[STARTUP_HEADERS] = startup_headers
        self.labelBindings = {}
        for seq in self.alignment.seqs:
            self.labelBindings[seq] = {
                '<Enter>': lambda e, s=seq:
                    self.sv.status(self.seqInfoText(s)),
                '<Double-Button>': lambda e, s=seq: self.sv._editSeqName(s)
            }
        for line in self.headers:
            self.labelBindings[line] = {}

        # first, set residue coloring to a known safe value...
        from clustalX import clustalInfo
        if single_sequence:
            prefResColor = RC_BLACK
        else:
            prefResColor = self.sv.prefs[RESIDUE_COLORING]
        if prefResColor == RC_BLACK:
            rc = self._cfBlack
        elif prefResColor == RC_RIBBON:
            from MAViewer import MOD_ASSOC
            self._residueHandlers = [chimera.triggers.addHandler(
                    'Residue', self._resChangeCB, None),
                self.sv.triggers.addHandler(MOD_ASSOC,
                    self._associationsCB, None)]
            rc = self._cfRibbon
        else:
            rc = self._cfClustalX
        for seq in self.alignment.seqs:
            seq.color_func = rc
        self._clustalXcache = {}
        self._clustalCategories, self._clustalColorings = clustalInfo()
        # try to set to external color scheme
        if prefResColor not in nonFileResidueColorings:
            try:
                self._clustalCategories, self._clustalColorings\
                        = clustalInfo(prefResColor)
            except Exception:
                schemes = self.sv.prefs[RC_CUSTOM_SCHEMES]
                if prefResColor in schemes:
                    schemes.remove(prefResColor)
                    self.sv.prefs[
                        RC_CUSTOM_SCHEMES] = schemes[:]
                self.sv.prefs[RESIDUE_COLORING] = RC_CLUSTALX
                from sys import exc_info
                replyobj.error("Error reading %s: %s\nUsing"
                    " default ClustalX coloring instead\n"
                    % (prefResColor, exc_info()[1]))
        """
        initial_headers = [hd for hd in self.alignment.headers if hd.shown]
        self.label_width = _find_label_width(self.alignment.seqs + initial_headers,
            self.sv.settings, self.font_metrics, self.emphasis_font_metrics, SeqBlock.label_pad)

        self._show_ruler = self.sv.settings.alignment_show_ruler_at_startup and len(self.alignment.seqs) > 1
        self.line_width = self.line_width_from_settings()
        self.numbering_widths = self.find_numbering_widths(self.line_width)
        """TODO
        self.showNumberings = [self.sv.leftNumberingVar.get(),
                    self.sv.rightNumberingVar.get()]
        self.lead_block = SeqBlock(self._labelCanvas(), self.mainCanvas,
            None, self.font, self.emphasis_font, self.font_metrics, self.emphasis_font_metrics,
            0, initialHeaders, self.alignment.seqs,
            self.line_width, self.labelBindings, lambda *args, **kw:
            self.sv.status(secondary=True, *args, **kw),
            self.show_ruler, self.treeBalloon, self.showNumberings,
            self.sv.settings)
        self._resizescrollregion()
        """
        label_scene = self._label_scene()
        from Qt.QtCore import Qt
        self.main_view.setAlignment(
            Qt.AlignCenter if label_scene == self.main_scene else Qt.AlignLeft)
        self.lead_block = SeqBlock(label_scene, self.main_scene, None, self.font,
            self.emphasis_font, self.font_metrics, self.emphasis_font_metrics, 0, initial_headers,
            self.alignment, self.line_width, {},
            lambda *args, **kw: self.sv.status(secondary=True, *args, **kw),
            self.show_ruler, None, self.show_numberings, self.sv.settings,
            self.label_width, self.font_pixels, self.numbering_widths, self.letter_gaps())
        self._update_scene_rects()

    def letter_gaps(self):
        column_sep_attr_name = "column_separation"
        if len(self.alignment.seqs) == 1:
            column_sep_attr_name = SINGLE_PREFIX + column_sep_attr_name
        return [getattr(self.sv.settings, column_sep_attr_name), 1]

    def _line_width_fits(self, pixels, num_characters):
        lnw, rnw = self.find_numbering_widths(num_characters)
        return (self.label_width + lnw
            + num_characters * (self.font_pixels[0] + self.letter_gaps()[0]) + rnw) < pixels

    def line_width_from_settings(self):
        if self.wrap_okay():
            # return a narrow line width the first time, so that the initial layout doesn't
            # force the tool column to be wider than needed
            if self._initial_layout:
                self._initial_layout = False
                return 20
            if len(self.alignment.seqs) == 1:
                prefix = SINGLE_PREFIX
            else:
                prefix = ""
            lwm = getattr(self.sv.settings, prefix + "line_width_multiple")
            lw = lwm
            try_lw = lw + lwm
            win_width = self.main_view.viewport().size().width()
            aln_len = len(self.alignment.seqs[0])
            while try_lw - lwm < aln_len \
            and self._line_width_fits(win_width, min(aln_len, try_lw)):
                lw = try_lw
                try_lw += lwm
            return lw
        # lay out entire sequence horizontally
        return 2 * len(self.alignment.seqs[0])

    @property
    def lines(self):
        return [hdr for hdr in self.alignment.headers if hdr.shown] + self.alignment.seqs

    """TODO
    def _molChange(self, trigger, myData, changes):
        # molecule attributes changed

        # find sequences (if any) with associations to changed mols
        assocSeqs = []
        for mol in changes.created | changes.modified:
            try:
                seq = self.sv.associations[mol]
            except KeyError:
                continue
            if seq not in assocSeqs:
                assocSeqs.append(seq)
        if assocSeqs:
            self.lead_block._molChange(assocSeqs)

    def _multiScroll(self, *args):
        self.labelCanvas.yview(*args)
        self.mainCanvas.yview(*args)

    def _newFont(self):
        if len(self.sv.seqs) == 1:
            prefix = SINGLE_PREFIX
        else:
            prefix = ""
        fontname, fontsize = (self.sv.prefs[prefix + FONT_NAME],
                        self.sv.prefs[prefix + FONT_SIZE])
        self.font = tkFont.Font(self.mainCanvas, (fontname, fontsize))
        self.sv.status("Changing to %d point %s"
                    % (fontsize, fontname), blankAfter=0)
        self.lead_block.fontChange(self.font)
        self.refreshTree()
        self.sv.region_browser.redraw_regions()
        self.sv.status("Font changed")

    def _newWrap(self):
        '''alignment wrapping preferences have changed'''
        line_width = self.line_width_from_settings()
        if line_width == self.line_width:
            return
        self.line_width = line_width
        self._reformat()

    def _pageDownCB(self, event):
        numBlocks = self.lead_block.numBlocks()
        v1, v2 = self.mainCanvas.yview()
        for i in range(numBlocks-1):
            if (i + 0.1) / numBlocks > v1:
                self.mainCanvas.yview_moveto(
                            float(i+1) / numBlocks)
                self.labelCanvas.yview_moveto(
                            float(i+1) / numBlocks)
                return
        self.mainCanvas.yview_moveto(float(numBlocks - 1) / numBlocks)
        self.labelCanvas.yview_moveto(float(numBlocks - 1) / numBlocks)

    def _pageUpCB(self, event):
        numBlocks = self.lead_block.numBlocks()
        v1, v2 = self.mainCanvas.yview()
        for i in range(numBlocks):
            if (i + 0.1) / numBlocks >= v1:
                if i == 0:
                    self.mainCanvas.yview_moveto(0.0)
                    self.labelCanvas.yview_moveto(0.0)
                    return
                self.mainCanvas.yview_moveto(
                            float(i-1) / numBlocks)
                self.labelCanvas.yview_moveto(
                            float(i-1) / numBlocks)

                return
        self.mainCanvas.yview_moveto(float(numBlocks - 1) / numBlocks)
        self.labelCanvas.yview_moveto(float(numBlocks - 1) / numBlocks)
    
    def realign(self, seqs, handleRegions=True):
        rb = self.sv.region_browser
        if handleRegions:
            # do what we can; move single-seq regions that begin and end
            # over non-gap characters, delete others
            deleteRegions = []
            regionUpdateInfo = []
            for region in rb.regions:
                if not region.blocks:
                    continue
                seq = region.blocks[0][0]
                if seq not in self.sv.seqs:
                    deleteRegions.append(region)
                    continue
                for block in region.blocks:
                    if block[0] != seq or block[1] != seq:
                        deleteRegions.append(region)
                        break
                else:
                    newBlocks = []
                    for block in region.blocks:
                        start, end = [seq.gapped2ungapped(x)
                            for x in block[2:]]
                        if start is None or end is None:
                            deleteRegions.append(region)
                            break
                        newBlocks.append((start, end))
                    else:
                        regionUpdateInfo.append((region, seq, newBlocks))
                        region.clear()
            for region in deleteRegions:
                rb.deleteRegion(region,
                    rebuildTable=(region==deleteRegions[-1]))
        savedSNs = self.showNumberings[:]
        if self.showNumberings[0]:
            self.setLeftNumberingDisplay(False)
        if self.showNumberings[1]:
            self.setRightNumberingDisplay(False)
        prevLen = len(self.alignment.seqs[0])
        for i in range(len(seqs)):
            self.alignment.seqs[i][:] = seqs[i]
        for header in self.headers:
            header.reevaluate()
        self._clustalXcache = {}
        self.lead_block.realign(prevLen)
        if savedSNs[0]:
            self.setLeftNumberingDisplay(True)
        if savedSNs[1]:
            self.setRightNumberingDisplay(True)
        self._resizescrollregion()
        if len(self.alignment.seqs[0]) != prevLen:
            self._recomputeScrollers()
        if handleRegions:
            for region, seq, ungappedBlocks in regionUpdateInfo:
                blocks = []
                for ub in ungappedBlocks:
                    gb = [seq.ungapped2gapped(x) for x in ub]
                    blocks.append([seq, seq] + gb)
                region.addBlocks(blocks)

    def recolor(self, seq):
        self.lead_block.recolor(seq)

    def _recomputeScrollers(self, width=None, height=None, xShowAt=None):
        if width is None:
            width = int(self.mainCanvas.cget('width'))
            height = int(self.mainCanvas.cget('height'))
        x1, y1, x2, y2 = map(int,
                self.mainCanvas.cget('scrollregion').split())

        needXScroll = x2 - x1 > width
        if needXScroll:
            if not self._hscrollMapped:
                self.horizScroll.grid(row=2, column=2,
                                sticky="ew")
                self._hscrollMapped = True
            if xShowAt is not None:
                self.mainCanvas.xview_moveto(xShowAt)
        elif not needXScroll and self._hscrollMapped:
            self.horizScroll.grid_forget()
            self._hscrollMapped = False

        if height > 50:
            needYScroll = y2 - y1 > height
        else:
            # avoid infinite scroller recursion
            needYScroll = False
            left, top, right, bottom = self.mainCanvas.bbox("all")
            vm = self.viewMargin
            newHeight = bottom - top + 2 * self.viewMargin
            self.mainCanvas.configure(height=newHeight)
            if self._labelCanvas(grid=0) == self.labelCanvas:
                self.labelCanvas.configure(height=newHeight)

        if needYScroll and not self._vscrollMapped:
            self.vertScroll.grid(row=1, column=3, sticky="ns")
            self._vscrollMapped = True
        elif not needYScroll and self._vscrollMapped:
            self.vertScroll.grid_forget()
            self._vscrollMapped = False

        if not self.wrap_okay() and self._vscrollMapped:
            self._hdivider.grid(row=2, column=0, sticky="new")
        else:
            self._hdivider.grid_forget()
    """

    def _reformat(self, cull_empty=False):
        self.sv.status("Reformatting alignment; please wait...", blank_after=0)
        """TODO
        if self.tree:
            activeNode = self.activeNode()
        """
        self.lead_block.destroy()
        initial_headers = [hd for hd in self.alignment.headers if hd.shown]
        self.label_width = _find_label_width(self.alignment.seqs + initial_headers,
            self.sv.settings, self.font_metrics, self.emphasis_font_metrics, SeqBlock.label_pad)
        self.line_width = self.line_width_from_settings()
        self.numbering_widths = self.find_numbering_widths(self.line_width)
        label_scene = self._label_scene()
        from Qt.QtCore import Qt
        self.main_view.setAlignment(
            Qt.AlignCenter if label_scene == self.main_scene else Qt.AlignLeft)
        self.lead_block = SeqBlock(label_scene, self.main_scene,
            None, self.font, self.emphasis_font, self.font_metrics, self.emphasis_font_metrics, 0,
            initial_headers, self.alignment, self.line_width, {},
            lambda *args, **kw: self.sv.status(secondary=True, *args, **kw),
            self.show_ruler, None, self.show_numberings, self.sv.settings,
            self.label_width, self.font_pixels, self.numbering_widths, self.letter_gaps())
        self._update_scene_rects()
        """TODO
        if self.tree:
            if self.treeShown:
                self.lead_block.showTree({'tree': self.tree},
                    self._treeCallback, self.nodesShown, active=activeNode)
            else:
                self.lead_block.treeNodeMap = {'active': activeNode }
        """
        self.sv.region_browser.redraw_regions(cull_empty=cull_empty)
        self.main_scene.update()
        self.label_scene.update()
        self._update_scene_rects()
        """TODO
        if len(self.alignment.seqs) != len(self._checkPoints[0]):
            self._checkPoint(fromScratch=True)
        else:
            self._checkPoint(checkChange=True)
        """
        self.sv.status("Alignment reformatted")

    def alignment_notification(self, note_name, note_data):
        if hasattr(self, 'lead_block'):
            if note_name == self.alignment.NOTE_REF_SEQ:
                self.lead_block.rerule()
            if note_name not in (self.alignment.NOTE_HDR_SHOWN, self.alignment.NOTE_HDR_VALUES,
                    self.alignment.NOTE_HDR_NAME):
                return
            if type(note_data) == tuple:
                hdr, bounds = note_data
            else:
                hdr = note_data
            if note_name == self.alignment.NOTE_HDR_SHOWN:
                if hdr.shown:
                    self.show_header(hdr)
                else:
                    self.hide_header(hdr)
            elif hdr.shown:
                if note_name == self.alignment.NOTE_HDR_VALUES:
                    if bounds is None:
                        bounds = (0, len(hdr)-1)
                    self.lead_block.refresh(hdr, *bounds)
                    self.main_scene.update()
                elif note_name == self.alignment.NOTE_HDR_NAME:
                    if self.label_width == _find_label_width(self.alignment.seqs +
                            [hdr for hdr in self.alignment.headers if hdr.shown], self.sv.settings,
                            self.font_metrics, self.emphasis_font_metrics, SeqBlock.label_pad):
                        self.lead_block.replace_label(hdr)
                        self.label_scene.update()
                    else:
                        self._reformat()

    def refresh(self, seq, left=0, right=None, update_attrs=True):
        if seq in self.alignment.headers and not seq.shown:
            return
        if right is None:
            right = len(self.alignment.seqs[0])-1
        self.lead_block.refresh(seq, left, right)
        self.main_scene.update()
        """TODO
        if update_attrs:
            self.sv.setResidueAttrs()
        """

    """TODO
    def refreshTree(self):
        if self.treeShown:
            self.lead_block.showTree({'tree': self.tree},
                    self._treeCallback, self.nodesShown,
                    active=self.activeNode())

    def _resChangeCB(self, trigName, myData, trigData):
        mols = set([r.molecule for r in trigData.modified])
        for m in mols:
            if m in self.sv.associations:
                self.recolor(self.sv.associations[m])
    """

    """TODO
    def _resizescrollregion(self):
        left, top, right, bottom = self.mainCanvas.bbox("all")
            vm = self.viewMargin
            left -= vm
            top -= vm
            right += vm
            bottom += vm
            if self._labelCanvas(grid=0) == self.labelCanvas:
                lbbox = self.labelCanvas.bbox("all")
                if lbbox is not None:
                    ll, lt, lr, lb = lbbox
                    ll -= vm
                    lt -= vm
                    lr += vm
                    lb += vm
                    top = min(top, lt)
                    bottom = max(bottom, lb)
                    self.labelCanvas.configure(width=lr-ll,
                            scrollregion=(ll, top, lr, bottom))
            self.mainCanvas.configure(scrollregion=
                            (left, top, right, bottom))
    """

    def restore_state(self, session, state):
        '''Used to restore header state, now done by alignment'''
        pass

    """TODO
    def saveEPS(self, fileName, colorMode, rotate, extent, hideNodes):
        if self.tree:
            savedNodeDisplay = self.nodesShown
            self.showNodes(not hideNodes)
        mainFileName = fileName
        msg = ""
        twoCanvas = self._labelCanvas(grid=0) != self.mainCanvas
        import os.path
        if twoCanvas:
            twoCanvas = True
            if fileName.endswith(".eps"):
                base = fileName[:-4]
                tail = ".eps"
            else:
                base = fileName
                tail = ""
            mainFileName = base + "-alignment" + tail
            labelFileName = base + "-names" + tail
            msg = "Sequence names and main alignment saved" \
                " as separate files\nSequence names saved" \
                " to %s\n" % os.path.split(labelFileName)[1]

        msg += "Alignment saved to %s\n" % os.path.split(
                            mainFileName)[1]
        mainKw = labelKw = {}
        if extent == "all":
            left, top, right, bottom = self.mainCanvas.bbox("all")
            if twoCanvas:
                ll, lt, lr, lb = self.labelCanvas.bbox("all")
                top = min(top, lt)
                bottom = max(bottom, lb)
                labelKw = {
                    'x': ll, 'y': top,
                    'width': lr-ll, 'height': bottom-top
                }
            mainKw = {
                'x': left, 'y': top,
                'width': right-left, 'height': bottom-top
            }
        self.mainCanvas.postscript(colormode=colorMode,
                file=mainFileName, rotate=rotate, **mainKw)
        if twoCanvas:
            self.labelCanvas.postscript(colormode=colorMode,
                file=labelFileName, rotate=rotate, **labelKw)
        if self.tree:
            self.showNodes(savedNodeDisplay)
        self.sv.status(msg)
    """

    def show_header(self, header):
        self.lead_block.show_header(header)
        self.sv.region_browser.redraw_regions()
        self._update_scene_rects()

    @property
    def show_left_numbering(self):
        return self.show_numberings[0]

    @show_left_numbering.setter
    def show_left_numbering(self, show):
        if self.show_numberings[0] == show:
            return
        self.show_numberings[0] = show
        new_widths = self.find_numbering_widths(self.line_width)
        if show:
            self.numbering_widths[:] = new_widths
        self.lead_block.show_left_numbering(show)
        if not show:
            self.numbering_widths[:] = new_widths
        self.sv.region_browser.redraw_regions()
        self._update_scene_rects()

    @property
    def show_right_numbering(self):
        return self.show_numberings[1]

    @show_right_numbering.setter
    def show_right_numbering(self, show):
        if self.show_numberings[1] == show:
            return
        self.show_numberings[1] = show
        new_widths = self.find_numbering_widths(self.line_width)
        if show:
            self.numbering_widths[:] = new_widths
        self.lead_block.show_right_numbering(show)
        if not show:
            self.numbering_widths[:] = new_widths
        self._update_scene_rects()

    @property
    def show_ruler(self):
        return self._show_ruler

    @show_ruler.setter
    def show_ruler(self, show_ruler):
        if show_ruler == self._show_ruler:
            return
        self._show_ruler = show_ruler
        self.lead_block.set_ruler_display(show_ruler)
        self.sv.region_browser.redraw_regions()
        self._update_scene_rects()

    def state(self):
        '''This used to save header state, now done by alignment'''
        return {}

    """TODO
    def seeBlocks(self, blocks):
        '''scroll canvas to show given blocks'''
        minx, miny, maxx, maxy = self.bbox_list(cover_gaps=True,
                                *blocks[0])[0]
        for block in blocks:
            for x1, y1, x2, y2 in self.bbox_list(cover_gaps=True,
                                *block):
                minx = min(minx, x1)
                miny = min(miny, y1)
                maxx = max(maxx, x2)
                maxy = max(maxy, y2)
        viewWidth = float(self.mainCanvas.cget('width'))
        viewHeight = float(self.mainCanvas.cget('height'))
        if maxx - minx > viewWidth or maxy - miny > viewHeight:
            # blocks don't fit in view; just show first block
            minx, miny, maxx, maxy = self.bbox_list(cover_gaps=True,
                                *blocks[0])[0]
        cx = (minx + maxx) / 2
        cy = (miny + maxy) / 2
        
        x1, y1, x2, y2 = map(int,
            self.mainCanvas.cget('scrollregion').split())
        totalWidth = float(x2 - x1 + 1)
        totalHeight = float(y2 - y1 + 1)

        if cx < x1 + viewWidth/2:
            cx = x1 + viewWidth/2
        if cy < y1 + viewHeight/2:
            cy = y1 + viewHeight/2
        startx = max(0.0, min((cx - viewWidth/2 - x1) / totalWidth,
                    (x2 - viewWidth - x1) / totalWidth))
        self.mainCanvas.xview_moveto(startx)
        starty = max(0.0, min((cy - viewHeight/2 - y1) / totalHeight,
                    (y2 - viewHeight - y1) / totalHeight))
        if not self.wrap_okay():
            self.labelCanvas.yview_moveto(starty)
        self.mainCanvas.yview_moveto(starty)

    def seeSeq(self, seq, highlightName):
        '''scroll up/down to center given seq, and possibly highlight name'''
        minx, miny, maxx, maxy = self.bbox_list(seq, seq, 0, 0)[0]
        viewHeight = float(self.mainCanvas.cget('height'))
        cy = (miny + maxy) / 2
        
        x1, y1, x2, y2 = map(int,
            self.mainCanvas.cget('scrollregion').split())
        totalHeight = float(y2 - y1 + 1)

        if cy < y1 + viewHeight/2:
            cy = y1 + viewHeight/2
        starty = max(0.0, min((cy - viewHeight/2 - y1) / totalHeight,
                    (y2 - viewHeight - y1) / totalHeight))
        if not self.wrap_okay():
            self.labelCanvas.yview_moveto(starty)
        self.mainCanvas.yview_moveto(starty)
        if highlightName:
            self.lead_block.highlightName(seq)

    def setClustalParams(self, categories, colorings):
        self._clustalXcache = {}
        self._clustalCategories, self._clustalColorings = \
                        categories, colorings
        if self.sv.prefs[RESIDUE_COLORING] in [RC_BLACK, RC_RIBBON]:
            return
        for seq in self.alignment.seqs:
            self.refresh(seq)

    def seqInfoText(self, aseq):
        basicText = "%s (#%d of %d; %d non-gap residues)\n" % (aseq.name,
            self.sv.seqs.index(aseq)+1, len(self.sv.seqs),
            len(aseq.ungapped()))
        if self.sv.intrinsicStructure \
                or not hasattr(aseq, 'matchMaps') or not aseq.matchMaps:
            return basicText
        return "%s%s associated with %s\n" % (basicText,
            _seq_name(aseq, self.sv.prefs),
            ", ".join(["%s (%s %s)" % (m.oslIdent(), m.name,
            aseq.matchMaps[m]['mseq'].name)
            for m in aseq.matchMaps.keys()]))

    def setColorFunc(self, coloring):
        if self._residueHandlers:
            chimera.triggers.deleteHandler('Residue',
                        self._residueHandlers[0])
            from MAViewer import MOD_ASSOC
            self.sv.triggers.deleteHandler(MOD_ASSOC,
                        self._residueHandlers[1])
            self._residueHandlers = None
        if coloring == RC_BLACK:
            cf = self._cfBlack
        elif coloring == RC_RIBBON:
            from MAViewer import MOD_ASSOC
            self._residueHandlers = [chimera.triggers.addHandler(
                    'Residue', self._resChangeCB, None),
                self.sv.triggers.addHandler(MOD_ASSOC,
                    self._associationsCB, None)]
            cf = self._cfRibbon
        else:
            cf = self._cfClustalX
        for seq in self.alignment.seqs:
            if not hasattr(seq, 'color_func'):
                seq.color_func = None
            if seq.color_func != cf:
                seq.color_func = cf
                self.recolor(seq)

    def _cfBlack(self, line, offset):
        return 'black'

    def _cfClustalX(self, line, offset):
        consensusChars = self.clustalConsensusChars(offset)
        res = line[offset].upper()
        if res in self._clustalColorings:
            for color, needed in self._clustalColorings[res]:
                if not needed:
                    return color
                for n in needed:
                    if n in consensusChars:
                        return color
        return 'black'

    def _cfRibbon(self, line, offset):
        if not hasattr(line, 'matchMaps') or not line.matchMaps:
            return 'black'
        ungapped = line.gapped2ungapped(offset)
        if ungapped == None:
            return 'black'
        rgbas = []
        for matchMap in line.matchMaps.values():
            try:
                r = matchMap[ungapped]
            except KeyError:
                continue
            rc = r.ribbonColor
            if rc == None:
                rc = r.molecule.color
            rgbas.append(rc.rgba())
        if not rgbas:
            return 'black'
        import numpy
        rgba = numpy.array(rgbas).mean(0)
        from CGLtk.color import rgba2tk
        return rgba2tk(rgba)

    def clustalConsensusChars(self, offset):
        try:
            consensusChars = self._clustalXcache[offset]
            return consensusChars
        except KeyError:
            pass
        chars = {}
        for seq in self.alignment.seqs:
            char = seq[offset].lower()
            chars[char] = chars.get(char, 0) + 1
        consensusChars = {}
        numSeqs = float(len(self.alignment.seqs))

        for members, threshold, result in self._clustalCategories:
            sum = 0
            for c in members:
                sum += chars.get(c, 0)
            if sum / numSeqs >= threshold:
                consensusChars[result] = True

        self._clustalXcache[offset] = consensusChars
        return consensusChars
        """

    def viewport_resized(self):
        self._resize_timer.stop()
        if self.line_width != self.line_width_from_settings():
            self._resize_timer.start()

    def wrap_okay(self):
        return _wrap_okay(len(self.alignment.seqs), self.sv.settings)

    """TODO
    def showNodes(self, show):
        if show == self.nodesShown:
            return
        self.nodesShown = show
        self.lead_block.showNodes(show)

    def showTree(self, show):
        if show == self.treeShown or not self.tree:
            return

        if show:
            self.lead_block.showTree({'tree': self.tree},
                        self._treeCallback, True,
                        active=self.activeNode())
            self.sv.triggers.activateTrigger(
                            DISPLAY_TREE, self.tree)
        else:
            self.lead_block.showTree(None, None, None)
            self.sv.triggers.activateTrigger(DISPLAY_TREE, None)
        self._resizescrollregion()
        self._recomputeScrollers(xShowAt=0.0)
        self.treeShown = show

    def updateNumberings(self):
        self.lead_block.updateNumberings()
        self._resizescrollregion()

    def usePhyloTree(self, tree, callback=None):
        treeInfo = {}
        if tree:
            tree.assignYpositions()
            tree.assignXpositions(branchStyle="weighted")
            tree.assignXdeltas()
            treeInfo['tree'] = tree
        self.lead_block.showTree(treeInfo, callback, True)
        self.lead_block.activateNode(tree)
        self._resizescrollregion()
        self._recomputeScrollers(xShowAt=0.0)
        self.tree = tree
        self._treeCallback = callback
        self.treeShown = self.nodesShown = bool(tree)
        self.sv.triggers.activateTrigger(DISPLAY_TREE, tree)
        """

    def _label_scene(self, grid=True):
        if self.wrap_okay():
            label_scene = self.main_scene
            if grid:
                self.label_view.hide()
                #self._vdivider.hide()
        else:
            label_scene = self.label_scene
            if grid:
                self.label_view.show()
                #self._vdivider.show()
        return label_scene
            
    """
    def _undoRedo(self, undo):
        # up/down == redo/undo
        curOffset, curEdited, curLeft, curRight = self._checkPoints[
                        self._checkPointIndex][1]
        if undo:
            if self._checkPointIndex == 0:
                replyobj.error("Nothing to undo.\n")
                return
            self._checkPointIndex -= 1
        else:
            if self._checkPointIndex == len(self._checkPoints) - 1:
                replyobj.error("Nothing to redo.\n")
                return
            self._checkPointIndex += 1
        checkPoint, info = self._checkPoints[self._checkPointIndex]
        chkOffset, chkEdited, chkLeft, chkRight = info
        if undo:
            offset = 0 - curOffset
            left, right = curLeft, curRight
        else:
            offset = chkOffset
            left, right = chkLeft, chkRight
        self.sv._edited = chkEdited
        if len(checkPoint[0]) != len(self.alignment.seqs[0]):
            self.sv.status("Need to change number of columns in"
                " alignment to allow for requested change.\n"
                "Please wait...")
            self.sv.realign(checkPoint, offset=offset)
            self.sv.status("Columns changed")
            return
        for seq, chkSeq in zip(self.alignment.seqs, checkPoint):
            seq[:] = chkSeq
        self._editRefresh(self.alignment.seqs, left, right)
        """

    def _update_scene_rects(self):
        self.main_scene.setSceneRect(self.main_scene.itemsBoundingRect())
        if self.label_scene != self.main_scene:
            # For scrolling to work right, ensure that vertical
            # size of label_scene is the same as main_scene
            lbr = self.label_scene.itemsBoundingRect()
            mr = self.main_scene.sceneRect()
            self.label_scene.setSceneRect(lbr.x(), mr.y(), lbr.width(), mr.height())

class SeqBlock:
    from Qt.QtCore import Qt
    normal_label_color = Qt.black
    header_label_color = Qt.blue
    multi_assoc_color = Qt.darkGreen
    label_pad = 3
    from Qt.QtGui import QPen
    qt_no_pen = QPen(Qt.NoPen)

    def __init__(self, label_scene, main_scene, prev_block, font, emphasis_font, font_metrics,
            emphasis_font_metrics, seq_offset, headers, alignment, line_width, label_bindings,
            status_func, show_ruler, tree_balloon, show_numberings, settings, label_width,
            font_pixels, numbering_widths, letter_gaps):
        self.label_scene = label_scene
        self.main_scene = main_scene
        self.prev_block = prev_block
        self.alignment = alignment
        self.font = font
        self.emphasis_font = emphasis_font
        self.font_metrics = font_metrics
        self.emphasis_font_metrics = emphasis_font_metrics
        self.label_bindings = label_bindings
        self.status_func = status_func
        """TODO
        self._mouseID = None
        if len(seqs) == 1:
            prefPrefix = SINGLE_PREFIX
        else:
            prefPrefix = ""
        """
        self.settings = settings
        """TODO
        self.letter_gaps = [prefs[prefPrefix + COLUMN_SEP],
                        prefs[prefPrefix + LINE_SEP]]
        if prefs[prefPrefix + TEN_RES_GAP]:
            self.chunk_gap = 20
        else:
            self.chunk_gap = 0
        """
        self.chunk_gap = 0
        block_space_attr_name = "block_space"
        if len(self.alignment.seqs) == 1:
            block_space_attr_name = SINGLE_PREFIX + block_space_attr_name
        if getattr(settings, block_space_attr_name):
            self.block_gap = 15
        else:
            self.block_gap = 3
        self.show_ruler = show_ruler
        self.tree_balloon = tree_balloon
        self.show_numberings = show_numberings
        self.seq_offset = seq_offset
        self.line_width = line_width
        self.label_width = label_width
        self.font_pixels = font_pixels
        self.numbering_widths = numbering_widths
        self.letter_gaps = letter_gaps

        if prev_block:
            self.top_y = prev_block.bottom_y + self.block_gap
            self.label_width = prev_block.label_width
            self.font_pixels = prev_block.font_pixels
            self.lines = prev_block.lines
            self.line_index = prev_block.line_index
            self.numbering_widths = prev_block.numbering_widths
            self._brushes = prev_block._brushes
            self.multi_assoc_brush = prev_block.multi_assoc_brush
            self.multi_assoc_pen = prev_block.multi_assoc_pen
        else:
            self.top_y = 0
            self.line_index = {}
            lines = list(headers) + list(self.alignment.seqs)
            for i in range(len(lines)):
                self.line_index[lines[i]] = i
            self.lines = lines
            from Qt.QtGui import QBrush, QPen
            """TODO
            if prefs[prefPrefix + BOLD_ALIGNMENT]:
                self.font = self.emphasis_font
            """
            from Qt.QtCore import Qt
            self.multi_assoc_brush = QBrush(self.multi_assoc_color, Qt.NoBrush)
            self.multi_assoc_pen = QPen(QBrush(self.multi_assoc_color, Qt.SolidPattern),
                                        0, Qt.DashLine)
            self._brushes = {}
            # long sequences can cause deep recursion...
            import sys
            recur_limit = sys.getrecursionlimit()
            # seems to be a hidden factor of 4 between the recursion
            # limit and the actual stack depth (!)
            seq_len = len(alignment.seqs[0])
            if 4 * (100 + seq_len / line_width) > recur_limit:
                sys.setrecursionlimit(4 * int(100 + seq_len / line_width))
            from chimerax.atomic import get_triggers
            self.handler = get_triggers().add_handler('changes', self._changes_cb)
        self.bottom_y = self.top_y

        self.label_texts = {}
        self.label_rects = {}
        self.numbering_texts = {}
        self.line_items = {}
        self.item_aux_info = {}
        self.tree_items = { 'lines': [], 'boxes': [] }
        self.highlighted_name = None

        self.layout_ruler()
        self.layout_lines(headers, self.header_label_color)
        self.layout_lines(alignment.seqs, self.normal_label_color)

        if seq_offset + line_width >= len(alignment.seqs[0]):
            self.next_block = None
        else:
            self.next_block = SeqBlock(label_scene, main_scene, self, self.font, self.emphasis_font,
                self.font_metrics, self.emphasis_font_metrics, seq_offset + line_width, headers, alignment,
                line_width, label_bindings, status_func, show_ruler, tree_balloon,
                show_numberings, settings, label_width, font_pixels, numbering_widths, letter_gaps)

    """TODO
    def activateNode(self, node, callback=None,
                    fromPrev=False, fromNext=False):
        active = self.treeNodeMap['active']
        if active == node:
            return
        if active:
            self.label_scene.itemconfigure(
                    self.treeNodeMap[active], fill='black')
        self.label_scene.itemconfigure(
                    self.treeNodeMap[node], fill='red')
        self.treeNodeMap['active'] = node
        if not fromPrev and self.prev_block:
            self.prev_block.activateNode(node, callback,
                                fromNext=True)
        if not fromNext and self.next_block:
            self.next_block.activateNode(node, callback,
                                fromPrev=True)
        if callback and not fromPrev and not fromNext:
            callback(node)

    def addSeqs(self, seqs, pushDown=0):
        self.top_y += pushDown
        if self.prev_block:
            newLabelWidth = self.prev_block.label_width
            newNumberingWidths = self.prev_block.numbering_widths
            insertIndex = len(self.lines) - len(seqs)
        else:
            insertIndex = len(self.lines)
            self.lines.extend(seqs)
            for i, seq in enumerate(seqs):
                self.line_index[seq] = insertIndex + i
            newLabelWidth = self.find_label_width(self.font_metrics,
                            self.emphasis_font_metrics)
            newNumberingWidths = self.find_numbering_widths(self.line_width)
        labelChange = newLabelWidth - self.label_width
        self.label_width = newLabelWidth
        numberingChanges = [newNumberingWidths[i]
                - self.numbering_widths[i] for i in range(2)]
        self.numbering_widths = newNumberingWidths

        for ruler_text in self.ruler_texts:
            self.main_scene.move(ruler_text,
                labelChange + numberingChanges[0], pushDown)
        self.bottom_ruler_y += pushDown

        self._move_lines(self.lines[:insertIndex], labelChange,
                        numberingChanges[0], pushDown)

        for i, seq in enumerate(seqs):
            self._layout_line(seq, self.normal_label_color,
                        line_index=insertIndex+i, adding=True)
        push = len(seqs) * (self.font_pixels[1] + self.letter_gaps[1])
        pushDown += push
        self._move_lines(self.lines[insertIndex+len(seqs):],
                labelChange, numberingChanges[0], pushDown)
        self.bottom_y += pushDown
        if self.next_block:
            self.next_block.addSeqs(seqs, pushDown=pushDown)
    """

    def _assoc_res_bind(self, item, aseq, index):
        item.setToolTip(self._mouse_res_text(aseq, index))

    def assoc_mod(self, aseq):
        label_text = self.label_texts[aseq]
        name = _seq_name(aseq, self.settings)
        from Qt.QtGui import QFontMetrics
        first_width = QFontMetrics(label_text.font()).horizontalAdvance(name)
        label_text.setFont(self._label_font(aseq))
        diff = QFontMetrics(label_text.font()).horizontalAdvance(name) - first_width
        if diff:
            label_text.moveBy(-diff, 0.0)
        label_text.setToolTip(self._label_tip(aseq))
        associated = self.has_associated_structures(aseq)
        if associated:
            self._colorize_label(aseq)
        else:
            if aseq in self.label_rects:
                self.label_scene.removeItem(self.label_rects[aseq])
                del self.label_rects[aseq]
                from Qt.QtCore import Qt
                label_text.setBrush(Qt.black)
        line_items = self.line_items[aseq]
        for i in range(len(line_items)):
            item = line_items[i]
            if not item:
                continue
            self._assoc_res_bind(item, aseq, self.seq_offset+i)
        """
        if self._large_alignment():
            line_items = self.line_items[aseq]
            for i in range(len(line_items)):
                item = line_items[i]
                if not item:
                    continue
                if associated:
                    self._assoc_res_bind(item, aseq, self.seq_offset+i)
                else:
                    item.tagBind('<Enter>', "")
                    item.tagBind('<Leave>', "")
        """
        if self.next_block:
            self.next_block.assoc_mod(aseq)
        
    def base_layout_info(self):
        half_x = self.font_pixels[0] / 2
        left_rect_off = 0 - half_x
        right_rect_off = self.font_pixels[0] - half_x
        return half_x, left_rect_off, right_rect_off

    def bbox_list(self, line1, line2, pos1, pos2, cover_gaps):
        if pos1 >= self.seq_offset + self.line_width:
            return self.next_block.bbox_list(line1, line2, pos1, pos2, cover_gaps)
        left = max(pos1, self.seq_offset) - self.seq_offset
        right = min(pos2, self.seq_offset + self.line_width - 1) - self.seq_offset
        bboxes = []
        if cover_gaps:
            bboxes.append(self._box_corners(left,right,line1,line2))
        else:
            l1 = self.line_index[line1]
            l2 = self.line_index[line2]
            lmin = min(l1, l2)
            lmax = max(l1, l2)
            for line in self.lines[l1:l2+1]:
                l = None
                for lo in range(left, right+1):
                    if line.gapped_to_ungapped(lo + self.seq_offset) is None:
                        # gap
                        if l is not None:
                            bboxes.append(self._box_corners(l, lo-1, line, line))
                            l = None
                    else:
                        # not gap
                        if l is None:
                            l = lo
                if l is not None:
                    bboxes.append(self._box_corners(l, right, line, line))

        if pos2 >= self.seq_offset + self.line_width:
            bboxes.extend(self.next_block.bbox_list(line1, line2, pos1, pos2, cover_gaps))
        return bboxes

    def _box_corners(self, left, right, line1, line2):
        ulx = self._left_seqs_edge() + left * (
                self.letter_gaps[0] + self.font_pixels[0]) + int(left/10) * self.chunk_gap
        uly = self.bottom_ruler_y + self.letter_gaps[1] + self.line_index[line1] * (
                self.font_pixels[1] + self.letter_gaps[1])
        lrx = self._left_seqs_edge() - self.letter_gaps[0] + (right+1) * (
                self.letter_gaps[0] + self.font_pixels[0]) + int(right/10) * self.chunk_gap
        lry = self.bottom_ruler_y + (self.line_index[line2] + 1) * (
                self.font_pixels[1] + self.letter_gaps[1])
        sep_attr_name = "column_separation"
        if len(self.alignment.seqs) == 1:
            sep_attr_name = SINGLE_PREFIX + sep_attr_name
        sep = getattr(self.settings, sep_attr_name)
        if sep < -1:
            overlap = int(abs(sep) / 2)
            ulx += overlap
            lrx -= overlap
        return ulx, uly-1, lrx, lry-1

    def bounded_by(self, x1, y1, x2, y2, exclude_headers):
        end = self.bottom_y + self.block_gap
        if y1 > end and y2 > end:
            if self.next_block:
                return self.next_block.bounded_by(x1, y1, x2, y2, exclude_headers)
            else:
                return (None, None, None, None)
        rel_y1 = self.relative_y(y1)
        rel_y2 = self.relative_y(y2)
        if rel_y1 < rel_y2:
            hi_row = self.row_index(rel_y1, bound="top")
            low_row = self.row_index(rel_y2, bound="bottom")
        else:
            hi_row = self.row_index(rel_y2, bound="top")
            low_row = self.row_index(rel_y1, bound="bottom")
        if hi_row is None or low_row is None:
            return (None, None, None, None)
        if exclude_headers:
            num_headers = len(self.lines) - len(self.alignment.seqs)
            if hi_row < num_headers:
                hi_row = num_headers
                if hi_row > low_row:
                    return (None, None, None, None)

        if y1 <= end and y2 <= end:
            if y1 > self.bottom_y and y2 > self.bottom_y \
            or y1 <= self.bottom_ruler_y and y2 <= self.bottom_ruler_y:
                # entirely in the same block gap or ruler
                return (None, None, None, None)
            # both on this block; determine right and left...
            left_x = min(x1, x2)
            right_x = max(x1, x2)
            left_pos = self.pos(left_x, bound="left")
            right_pos = self.pos(right_x, bound="right")
        else:
            # the one on this block is left...
            if y1 <= end:
                left_x, right_x, lowY = x1, x2, y2
            else:
                left_x, right_x, lowY = x2, x1, y1
            left_pos = self.pos(left_x, bound="left")
            if self.next_block:
                right_pos = self.next_block.pos(right_x, bound="right", y=lowY)
            else:
                right_pos = self.pos(right_x, bound="right")
        if left_pos is None or right_pos is None or left_pos > right_pos:
            return (None, None, None, None)
        return (self.lines[hi_row], self.lines[low_row], left_pos, right_pos)


    def _brush(self, color):
        from Qt.QtGui import QBrush, QColor
        if not isinstance(color, QColor):
            color = QColor(color)
        rgb = color.rgb()
        try:
            return self._brushes[rgb]
        except KeyError:
            brush = QBrush(color)
            self._brushes[rgb] = brush
            return brush

    def _changes_cb(self, trigger_name, changes):
        reasons = changes.atom_reasons()
        if "color changed" not in reasons:
            return
        # allow alignment to update itself, so check on 'changes done' trigger
        def update_swatches(*args, self=self):
            # for performance reasons, changes.modified_atoms() returns nothing for color changes,
            # so just redo all label colors
            for aseq in self.alignment.seqs:
                assoc_structures = set([chain.structure for chain in aseq.match_maps.keys()])
                if not assoc_structures or len(assoc_structures) > 1:
                    continue
                block = self
                while block is not None:
                    block._colorize_label(aseq)
                    block = block.next_block
            from chimerax.core.triggerset import DEREGISTER
            return DEREGISTER
        from chimerax.atomic import get_triggers
        get_triggers().add_handler('changes done', update_swatches)

    def _color_func(self, line):
            if hasattr(line, 'position_color'):
                from .region_browser import get_rgba, rgba_to_qcolor
                return lambda l, o: rgba_to_qcolor(get_rgba(l.position_color(o)))
            from Qt.QtCore import Qt
            return lambda l, o, color=Qt.black: color
    """TODO
        try:
            return line.color_func
        except AttributeError:
            return lambda l, o: 'black'
    """

    def _colorize_label(self, aseq):
        label_text = self.label_texts[aseq]
        bbox = label_text.boundingRect()
        if aseq in self.label_rects:
            label_rect = self.label_rects[aseq]
        else:
            # Qt seems to make the bounding box enclose all of the inter-line space
            # below the text and none of that space above it.  Move the bounding box up
            # to make the enclosure look more even
            ## leading() doesn't seem to return anything useful
            ##from Qt.QtGui import QFontMetrics
            ##interline = QFontMetrics(label_text.font()).leading()
            interline = 2
            bbox.adjust(0, -interline/2, 0, -interline/2)
            label_rect = self.label_scene.addRect(label_text.mapRectToScene(bbox))
            label_rect.setZValue(-1)
            self.label_rects[aseq] = label_rect
        structures = set([chain.structure for chain in aseq.match_maps.keys()])
        from Qt.QtGui import QColor
        if len(structures) > 1:
            brush = self.multi_assoc_brush
            pen = self.multi_assoc_pen
            contrast = (0.0, 0.0, 0.0)
        else:
            import numpy
            if len(aseq.match_maps) == 1:
                chain = list(aseq.match_maps.keys())[0]
                colors = chain.existing_residues.existing_principal_atoms.colors
                if len(colors) == 0:
                    colors = chain.existing_residues.atoms.colors
                color = numpy.sum(colors, axis=0) / len(colors)
            else:
                struct = structures.pop()
                if struct.model_color is not None:
                    color = struct.model_color
                else:
                    colors = struct.atoms.colors
                    color = numpy.sum(colors, axis=0) / len(colors)
            from Qt.QtCore import Qt
            from Qt.QtGui import QPen, QBrush
            brush = QBrush(QColor(*color), Qt.SolidPattern)
            if 255 == color[0] == color[1] == color[2]:
                pen = QPen(QColor(216, 216, 216))
            else:
                pen = QPen(brush, 0, Qt.SolidLine)
            from chimerax.core.colors import contrast_with
            contrast = contrast_with([c/255.0 for c in color])
        label_rect.setBrush(brush)
        label_rect.setPen(pen)
        text_brush = label_text.brush()
        text_brush.setColor(QColor(*[int(c*255+0.5) for c in contrast]))
        label_text.setBrush(text_brush)

    def _compute_numbering(self, line, end):
        if end == 0:
            count = len([c for c in line[:self.seq_offset]
                        if c.isalpha() or c == '?'])
            if count == len(line.ungapped()):
                count -= 1
        else:
            count = len([c for c in line[:self.seq_offset
                + self.line_width] if c.isalpha()] or c == '?') - 1
        return line_numbering_start(line) + count

    """TODO
    def dehighlightName(self):
        if self.highlighted_name:
            self.label_scene.itemconfigure(self.highlighted_name,
                fill=self.normal_label_color)
            self.highlighted_name = None
            if self.next_block:
                self.next_block.dehighlightName()
    """

    def destroy(self):
        if not self.prev_block:
            self.handler.remove()
        if self.next_block:
            self.next_block.destroy()
            self.next_block = None
        for ruler_text in self.ruler_texts:
            self.main_scene.removeItem(ruler_text)
        for label_text in self.label_texts.values():
            self.label_scene.removeItem(label_text)
        for numberings in self.numbering_texts.values():
            for numbering in numberings:
                if numbering:
                    self.main_scene.removeItem(numbering)
        """TODO
        for box in self.tree_items['boxes']:
            self.tree_balloon.tagunbind(self.label_scene, box)
        for tree_items in self.tree_items.values():
            for treeItem in tree_items:
                self.label_scene.removeItem(treeItem)
        """
        for label_rect in self.label_rects.values():
            self.label_scene.removeItem(label_rect)
        for line_items in self.line_items.values():
            for line_item in line_items:
                if line_item is not None:
                    self.main_scene.removeItem(line_item)

    """TODO
    def fontChange(self, font, emphasis_font=None, pushDown=0):
        self.top_y += pushDown
        self.font = font
        if emphasis_font:
            self.emphasis_font = emphasis_font
        else:
            self.emphasis_font = self.font.copy()
            self.emphasis_font.configure(weight=tkFont.BOLD)
        if len(self.alignment.seqs) == 1:
            prefPrefix = SINGLE_PREFIX
        else:
            prefPrefix = ""
        if self.prefs[prefPrefix + BOLD_ALIGNMENT]:
            self.font = self.emphasis_font
        if self.prev_block:
            newLabelWidth = self.prev_block.label_width
            font_pixels = self.prev_block.font_pixels
            newNumberingWidths = self.prev_block.numbering_widths
        else:
            newLabelWidth = self.find_label_width(font,
                            self.emphasis_font)
            w, h = self.measureFont(self.font)
            font_pixels = (w+1, h+1) # allow padding
            newNumberingWidths = self.find_numbering_widths(self.line_width)
        labelChange = newLabelWidth - self.label_width
        leftNumberingChange = newNumberingWidths[0] \
                        - self.numbering_widths[0]
        curWidth, curHeight = self.font_pixels
        newWidth, newHeight = font_pixels

        perLine = newHeight - curHeight
        perChar = newWidth - curWidth

        over = labelChange + leftNumberingChange + perChar / 2
        down = pushDown + perLine
        for ruler_text in self.ruler_texts:
            self.main_scene.itemconfigure(ruler_text, font=font)
            self.main_scene.move(ruler_text, over, down)
            over += perChar * 10
        self.bottom_ruler_y += down

        down = pushDown + 2 * perLine
        for line in self.lines:
            label_text = self.label_texts[line]
            self.label_scene.itemconfigure(label_text,
                        font=self._label_font(line))
            self.label_scene.move(label_text, 0, down)
            if line in self.label_rects:
                self._colorizeLabel(line)
            leftNumberingText = self.numbering_texts[line][0]
            if leftNumberingText:
                self.main_scene.itemconfigure(leftNumberingText,
                                font=font)
                self.main_scene.move(leftNumberingText,
                    labelChange + leftNumberingChange, down)
            down += perLine

        down = pushDown + 2 * perLine
        histLeft = perChar / 2
        histRight = perChar - histLeft
        for seq in self.lines:
            over = labelChange + leftNumberingChange + perChar / 2
            line_items = self.line_items[seq]
            item_aux_info = self.item_aux_info[seq]
            color_func = self._color_func(seq)
            depictable = hasattr(seq, 'depiction_val')
            for i in range(len(line_items)):
                line_item = line_items[i]
                oldx, oldy = item_aux_info[i]
                newx, newy = item_aux_info[i] = (oldx + over, oldy + down)
                if line_item:
                    index = self.seq_offset + i
                    if depictable:
                        val = seq.depiction_val(index)
                    else:
                        val = seq[index]
                    if isinstance(val, str):
                        if len(val) > 1:
                            line_item.delete()
                            left = newx - newWidth/2.0
                            right = newx + newWidth/2.0
                            top = newy - newHeight
                            line_item.draw(val, left, right, top, newy,
                                            color_func(seq, index))
                        else:
                            line_item.move(over, down)
                            line_item.configure(font=self.font)
                    else:
                        leftX, top_y, rightX, bottom_y = line_item.coords()
                        leftX += over - histLeft
                        rightX += over + histRight
                        oldHeight = bottom_y - top_y
                        bottom_y += down
                        top_y = bottom_y - (newHeight * float(oldHeight)
                                / curHeight)
                        line_item.coords(leftX, top_y, rightX, bottom_y)

                over += perChar
            rightNumberingText = self.numbering_texts[seq][1]
            if rightNumberingText:
                self.main_scene.move(rightNumberingText, over,
                                    down)
                self.main_scene.itemconfigure(
                        rightNumberingText, font=font)
            down += perLine
        self.label_width = newLabelWidth
        self.font_pixels = font_pixels
        self.bottom_y += down
        self.numbering_widths = newNumberingWidths
        if self.next_block:
            self.next_block.fontChange(font,
                emphasis_font=self.emphasis_font, pushDown=down)
    """

    def _get_xs(self, amount):
        xs = []
        half_x, left_rect_off, right_rect_off = self.base_layout_info()
        x = self._left_seqs_edge() + half_x
        for chunk_start in range(0, amount, 10):
            for offset in range(chunk_start, min(chunk_start + 10, amount)):
                xs.append(x)
                x += self.font_pixels[0] + self.letter_gaps[0]
            x += self.chunk_gap
        return xs

    def has_associated_structures(self, line):
        if getattr(line, 'match_maps', None) \
        and [chain for chain in line.match_maps.keys() if not chain.structure.deleted]:
            return True
        return False

    def hide_header(self, header, push_down=0, del_index=None):
        self.top_y += push_down
        if self.prev_block:
            self.label_width = self.prev_block.label_width
        else:
            del_index = self.line_index[header]
            del self.lines[del_index]
            del self.line_index[header]
            for line in self.lines[del_index:]:
                self.line_index[line] -= 1
            self.label_width = _find_label_width(self.lines, self.settings, self.font_metrics,
                self.emphasis_font_metrics, self.label_pad)

        for ruler_text in self.ruler_texts:
            ruler_text.moveBy(0, push_down)
        self.bottom_ruler_y += push_down

        self._move_lines(self.lines[:del_index], 0, 0, push_down)

        label_text = self.label_texts[header]
        del self.label_texts[header]
        label_text.hide()
        self.label_scene.removeItem(label_text)

        line_items = self.line_items[header]
        del self.line_items[header]
        for item in line_items:
            if item is not None:
                item.hide()
                self.main_scene.removeItem(item)
        del self.item_aux_info[header]
        pull = self.font_pixels[1] + self.letter_gaps[1]
        push_down -= pull
        self._move_lines(self.lines[del_index:], 0, 0, push_down)
        self._move_tree(push_down)

        self.bottom_y += push_down
        if self.next_block:
            self.next_block.hide_header(header, push_down=push_down, del_index=del_index)

    """TODO
    def highlightName(self, line):
        if self.highlighted_name:
            self.label_scene.itemconfigure(self.highlighted_name,
                fill=self.normal_label_color)
        self.highlighted_name = text = self.label_texts[line]
        self.label_scene.itemconfigure(text, fill='red')
        if self.next_block:
            self.next_block.highlightName(line)
    """

    def _label_font(self, line):
        if self.has_associated_structures(line):
            return self.emphasis_font
        return self.font

    def _label_tip(self, line):
        if not isinstance(line, Sequence):
            return ""
        basic_text = "%s (#%d of %d; %d non-gap residues)" % (line.name,
            self.alignment.seqs.index(line)+1, len(self.alignment.seqs), len(line.ungapped()))
        if not line.match_maps:
            return basic_text
        return "%s\n%s associated with:\n%s" % (basic_text, _seq_name(line, self.settings),
            "\n".join(["#%s (%s %s)" % (m.structure.id_string, m.structure.name,
            line.match_maps[m].struct_seq.name) for m in line.match_maps.keys()]))

    def _large_alignment(self):
        # for now, return False until performance can be tested
        return False
        return len(self.alignment.seqs) * len(self.alignment.seqs[0]) >= 250000

    def layout_ruler(self, rerule=False):
        if rerule:
            for text in self.ruler_texts:
                text.hide()
                self.main_scene.removeItem(text)
        self.ruler_texts = []
        if not self.show_ruler:
            self.bottom_ruler_y = self.top_y
            return
        x = self._left_seqs_edge() + self.font_pixels[0]/2
        y = self.top_y + self.font_pixels[1] + self.letter_gaps[1]

        end = min(self.seq_offset + self.line_width, len(self.alignment.seqs[0]))
        ref_seq = self.alignment.reference_seq
        for chunk_start in range(self.seq_offset, end, 10):
            if ref_seq is None:
                ruler_text = "%d" % (chunk_start+1)
            else:
                index = ref_seq.gapped_to_ungapped(chunk_start)
                if index is None:
                    # in a gap in the reference sequence
                    for i in range(chunk_start-1, -1, -1):
                        left_index = ref_seq.gapped_to_ungapped(i)
                        if left_index is not None:
                            break
                    else:
                        left_index = None

                    for i in range(chunk_start+1, len(ref_seq.ungapped())):
                        right_index = ref_seq.gapped_to_ungapped(i)
                        if right_index is not None:
                            break
                    else:
                        right_index = None

                    if left_index is None:
                        if right_index is None:
                            ruler_text = "N/A"
                        else:
                            ruler_text = ("(<%d)" % (right_index+1))
                    elif right_index is None:
                        ruler_text = "(>%d)" % (left_index+1)
                    else:
                        ruler_text = "(%d/%d)" % (left_index+1, right_index+1)
                else:
                    ruler_text = "%d" % (index+1)
            text = self.main_scene.addSimpleText(ruler_text, font=self.font)
            # anchor='s': subtract the height and half the width
            rect = text.sceneBoundingRect()
            text.setPos(x - rect.width()/2, y - rect.height())
            self.ruler_texts.append(text)
            x += self.chunk_gap + 10 * (self.font_pixels[0] + self.letter_gaps[0])
        if not rerule:
            self.bottom_y += self.font_pixels[1] + self.letter_gaps[1]
            self.bottom_ruler_y = y

    def _layout_line(self, line, label_color, base_layout_info=None, end=None,
                            line_index=None, adding=False):
        if not end:
            end = min(self.seq_offset + self.line_width, len(self.alignment.seqs[0]))
        if base_layout_info:
            half_x, left_rect_off, right_rect_off = base_layout_info
        else:
            half_x, left_rect_off, right_rect_off = self.base_layout_info()

        x = 0
        if line_index is None:
            y = self.bottom_y + self.font_pixels[1] + self.letter_gaps[1]
        else:
            y = self.bottom_ruler_y + (line_index+1) * (self.font_pixels[1] + self.letter_gaps[1])

        text = self.label_scene.addSimpleText(_seq_name(line, self.settings),
            font=self._label_font(line))
        text.setBrush(self._brush(label_color))
        # anchor='sw': subtract the height
        rect = text.sceneBoundingRect()
        text.setPos(x, y - rect.height())
        # but then right justify
        text.moveBy(self.label_width - self.label_pad - rect.width(), 0)
        self.label_texts[line] = text
        text.setToolTip(self._label_tip(line))
        if self.has_associated_structures(line):
            self._colorize_label(line)
        """TODO
        bindings = self.label_bindings[line]
        if bindings:
            for eventType, function in bindings.items():
                self.label_scene.tag_bind(text,
                            eventType, function)
        """
        color_func = self._color_func(line)
        line_items = []
        item_aux_info = []
        xs = self._get_xs(end - self.seq_offset)
        if self._large_alignment():
            res_status = hasattr(line, "match_maps") and line.match_maps
        else:
            res_status = line in self.alignment.seqs or adding
        for i in range(end - self.seq_offset):
            item = self.make_item(line, self.seq_offset + i, xs[i],
                y, half_x, left_rect_off, right_rect_off, color_func)
            if res_status:
                self._assoc_res_bind(item, line, self.seq_offset + i)
            line_items.append(item)
            item_aux_info.append((xs[i], y))

        self.line_items[line] = line_items
        self.item_aux_info[line] = item_aux_info
        if line_index is None:
            self.bottom_y += self.font_pixels[1] + self.letter_gaps[1]

        numberings = [None, None]
        if line_numbering_start(line) is not None:
            for numbering in range(2):
                if self.show_numberings[numbering]:
                    numberings[numbering] = self._make_numbering(line, numbering)
        self.numbering_texts[line] = numberings

    def layout_lines(self, lines, label_color):
        end = min(self.seq_offset + self.line_width, len(self.alignment.seqs[0]))
        bli = self.base_layout_info()
        for line in lines:
            self._layout_line(line, label_color, bli, end)

    """TODO
    def _layoutTree(self, treeInfo, node, callback, nodesShown,
                            prevXpos=None):
        def xFunc(x, delta):
            return 216.0 * (x - 1.0 + delta/100.0)
        def yFunc(y):
            return (self.bottom_ruler_y + self.letter_gaps[1] +
                0.5 * self.font_pixels[1] +
                (y + len(self.lines) - len(self.alignment.seqs)) *
                (self.font_pixels[1] + self.letter_gaps[1]))
        x = xFunc(node.xPos, node.xDelta)
        y = yFunc(node.yPos)
        lines = self.tree_items['lines']
        if prevXpos is not None:
            lines.append(self.label_scene.create_line(
                    "%.2f" % prevXpos, y, "%.2f" % x, y))
        if not node.subNodes:
            rightmostX = xFunc(1.0, 0.0)
            if x != rightmostX:
                lines.append(self.label_scene.create_line("%.2f" % x, y,
                    "%.2f" % rightmostX, y, dash=[1,3]))
            return
        subYs = [n.yPos for n in node.subNodes]
        minSubY = min(subYs)
        maxSubY = max(subYs)
        lines.append(self.label_scene.create_line(
            "%.2f" % x, yFunc(minSubY), "%.2f" % x, yFunc(maxSubY)))
        if self.treeNodeMap['active'] == node:
            fill = 'red'
        else:
            fill = 'black'
        box = self.label_scene.create_rectangle(x-2, y-2,
                    x+2, y+2, fill=fill, outline="black")
        self.label_scene.tag_bind(box, "<ButtonRelease-1>",
            lambda e, cb=callback, n=node: self.activateNode(n, cb))
        self.treeNodeMap[node] = box
        if node.label:
            balloonText = "%s: " % node.label
        else:
            balloonText = ""
        balloonText += "%d sequences" % node.countNodes("leaf")
        self.tree_balloon.tagbind(self.label_scene, box, balloonText)
        self.tree_items['boxes'].append(box)
        if not nodesShown:
            self.label_scene.itemconfigure(box, state='hidden')
        for sn in node.subNodes:
            self._layoutTree(treeInfo, sn, callback, nodesShown, x)
    """

    def _left_seqs_edge(self):
        if self.label_scene == self.main_scene:
            return self.label_width + self.letter_gaps[0] + self.numbering_widths[0]
        return 0

    def make_item(self, line, offset, x, y, half_x, left_rect_off, right_rect_off, color_func):
        if hasattr(line, 'depiction_val'):
            info = line.depiction_val(offset)
        else:
            info = line[offset]
        if isinstance(info, str):
            """TODO: try to use QGraphicsItem/QGraphicsItemGroup instead of high-overhead LineItem
            if len(info) > 1:
                left = x + left_rect_off
                right = x + right_rect_off
                bottom = y
                top = y - self.font_pixels[1]
                return LineItem(info, self.main_scene, left, right, top, bottom,
                    color_func(line, offset))
            """
            text = self.main_scene.addSimpleText(info, font=self.font)
            # anchor='s': subtract the height and half the width
            rect = text.sceneBoundingRect()
            text.setPos(x - rect.width()/2, y - rect.height())
            text.setBrush(self._brush(color_func(line, offset)))
            return text
        if info != None and info > 0.0:
            return self.main_scene.addRect(x + left_rect_off, y-1, right_rect_off - left_rect_off,
                -info * self.font_pixels[1], pen=self.qt_no_pen,
                brush=self._brush(color_func(line, offset)))
        return None

    def _make_numbering(self, line, numbering):
        n = self._compute_numbering(line, numbering)
        fmt = "%d " if numbering == 0 else " %d"
        item = self.main_scene.addSimpleText(fmt % n, font=self.font)
        x, y = self.item_aux_info[line][-1]
        rect = item.sceneBoundingRect()
        if numbering == 0:
            x = self._left_seqs_edge() - rect.width()
        else:
            x += self.base_layout_info()[0]
        item.setPos(x, y - rect.height())
        return item

    """TODO
    def measureFont(self, font):
        height = self.font.actual('size')
        if height > 0: # points
            height = self.main_scene.winfo_pixels(
                            "%gp" % float(height))
        else:
            height = 0 - height

        width = 0
        for let in string.uppercase:
            width = max(width, font.measure(let))
        return (width, height)

    def _molChange(self, seqs):
        for seq in seqs:
            self._colorizeLabel(seq)
        if self.next_block:
            self.next_block._molChange(seqs)
    """

    def _mouse_res_text(self, aseq, index):
        ungapped = aseq.gapped_to_ungapped(index)
        if ungapped is None:
            res_text = "gap"
        elif aseq.match_maps:
            residues = []
            for match_map in aseq.match_maps.values():
                try:
                    residues.append(match_map[ungapped])
                except KeyError:
                    continue
            if residues:
                if len(residues) < 20:
                    res_text = "\n".join([str(r) for r in residues])
                else:
                    res_text = "\n".join([str(r) for r in residues[:10]])
                    res_text += "\n and %d more..." % (len(residues) - 10)
            else:
                res_text = "no corresponding structure residue"
        else:
            if aseq.numbering_start is None:
                off = 1
            else:
                off = aseq.numbering_start
            res_text = "sequence position %d" % (ungapped+off)

        return res_text

    def _move_lines(self, lines, over_label, over_number, down):
        over = over_label + over_number
        for line in lines:
            self.label_texts[line].moveBy(0, down)

            lnum, rnum = self.numbering_texts[line]
            if lnum:
                lnum.moveBy(over_label, down)
            if rnum:
                rnum.moveBy(over, down)
            for item in self.line_items[line]:
                if item is not None:
                    item.moveBy(over, down)
            item_aux_info = []
            for oldx, oldy in self.item_aux_info[line]:
                item_aux_info.append((oldx+over, oldy+down))
            self.item_aux_info[line] = item_aux_info
            if line in self.label_rects:
                self.label_rects[line].moveBy(0, down)
    def _move_tree(self, down):
        for item_type, item_list in self.tree_items.items():
            for item in item_list:
                item.moveBy(0, down)

    """TODO
    def numBlocks(self):
        if self.next_block:
            return self.next_block.numBlocks() + 1
        return 1
    """

    def pos(self, x, bound=None, y=None):
        '''return 'sequence' position of x'''
        if y is not None and self.next_block and y > self.bottom_y + self.block_gap:
            return self.next_block.pos(x, bound, y)
        if x < self._left_seqs_edge():
            if bound == "left":
                return self.seq_offset
            elif bound == "right":
                if self.seq_offset > 0:
                    return self.seq_offset -1
                else:
                    return None
            else:
                return None
        chunk = int((x - self._left_seqs_edge()) /
            (10 * (self.font_pixels[0] + self.letter_gaps[0]) + self.chunk_gap))
        chunk_x = x - self._left_seqs_edge() - chunk * (
            10 * (self.font_pixels[0] + self.letter_gaps[0]) + self.chunk_gap)
        chunk_offset = int(chunk_x / (self.font_pixels[0] + self.letter_gaps[0]))
        offset = 10 * chunk + min(chunk_offset, 10)
        my_line_width = min(self.line_width, len(self.alignment.seqs[0]) - self.seq_offset)
        if offset >= my_line_width:
            if bound == "left":
                if self.next_block:
                    return self.seq_offset + my_line_width
                return None
            elif bound == "right":
                return self.seq_offset + my_line_width - 1
            return None
        offset = 10 * chunk + min(chunk_offset, 9)
        right_edge = self._left_seqs_edge() + \
            chunk * (10 * (self.font_pixels[0] + self.letter_gaps[0]) + self.chunk_gap) + \
            (chunk_offset + 1) * (self.font_pixels[0] + self.letter_gaps[0])
        if chunk_offset >= 10 or right_edge - x < self.letter_gaps[0]:
            # in gap
            if bound == "left":
                return self.seq_offset + offset + 1
            elif bound == "right":
                return self.seq_offset + offset
            return None
        # on letter
        return self.seq_offset + offset

    """TODO
    def realign(self, prevLen):
        '''sequences globally realigned'''

        if _wrap_okay(len(self.alignment.seqs), self.settings):
            blockEnd = self.seq_offset + self.line_width
            prev_blockLen = min(prevLen, blockEnd)
            curBlockLen = min(len(self.alignment.seqs[0]), blockEnd)
        else:
            blockEnd = len(self.alignment.seqs[0])
            self.line_width = blockEnd
            prev_blockLen = prevLen
            curBlockLen = blockEnd

        half_x, left_rect_off, right_rect_off = self.base_layout_info()

        numUnchanged = min(prev_blockLen, curBlockLen) - self.seq_offset
        for line in self.lines:
            line_items = self.line_items[line]
            item_aux_info = self.item_aux_info[line]
            color_func = self._color_func(line)
            if self._large_alignment():
                res_status = hasattr(line, "match_maps") and line.match_maps
            else:
                res_status = line in self.alignment.seqs
            for i in range(numUnchanged):
                item = line_items[i]
                if item is not None:
                    item.delete()
                x, y = item_aux_info[i]
                line_items[i] = self.make_item(line,
                    self.seq_offset + i, x, y, half_x,
                    left_rect_off, right_rect_off, color_func)
                if res_status:
                    self._assoc_res_bind(line_items[i], line, self.seq_offset + i)

        if curBlockLen < prev_blockLen:
            # delete excess items
            self.layout_ruler(rerule=True)
            for line in self.lines:
                line_items = self.line_items[line]
                for i in range(curBlockLen, prev_blockLen):
                    item = line_items[i - self.seq_offset]
                    if item is None:
                        continue
                    item.delete()
                start = curBlockLen - self.seq_offset
                end = prev_blockLen - self.seq_offset
                line_items[start:end] = []
                self.item_aux_info[line][start:end] = []
        elif curBlockLen > prev_blockLen:
            # add items
            self.layout_ruler(rerule=True)
            for line in self.lines:
                if self._large_alignment():
                    res_status = hasattr(line, "match_maps") and line.match_maps
                else:
                    res_status = line in self.alignment.seqs
                line_items = self.line_items[line]
                item_aux_info = self.item_aux_info[line]
                x, y = item_aux_info[0]
                color_func = self._color_func(line)
                xs = self._get_xs(curBlockLen - self.seq_offset)
                for i in range(prev_blockLen, curBlockLen):
                    x = xs[i - self.seq_offset]
                    line_items.append(self.make_item(line, i,
                        x, y, half_x, left_rect_off,
                        right_rect_off, color_func))
                    item_aux_info.append((x, y))
                    if res_status:
                        self._assoc_res_bind(line_items[-1], line, i)

        if len(self.alignment.seqs[0]) <= blockEnd:
            # no further blocks
            if self.next_block:
                self.next_block.destroy()
                self.next_block = None
        else:
            # more blocks
            if self.next_block:
                self.next_block.realign(prevLen)
            else:
                self.next_block = SeqBlock(self.label_scene, self.main_scene, self, self.font,
                    self.enphasis_font, self.font_metrics, self.emphasis_font_metrics,
                    self.seq_offset + self.line_width, self.lines[:0-len(self.alignment.seqs)],
                    self.alignment.seqs, self.line_width, self.label_bindings, self.status_func,
                    self.show_ruler, self.tree_balloon, self.show_numberings, self.settings,
                    self.label_width, self.font_pixels, self.numbering_widths, self.letter_gaps)

    def recolor(self, seq):
        if self.next_block:
            self.next_block.recolor(seq)

        color_func = self._color_func(seq)

        for i, line_item in enumerate(self.line_items[seq]):
            if line_item is None:
                continue
            line_item.configure(fill=color_func(seq, self.seq_offset + i))
    """

    def refresh(self, seq, left, right):
        if self.seq_offset + self.line_width <= right:
            self.next_block.refresh(seq, left, right)
        if left >= self.seq_offset + self.line_width:
            return
        my_left = max(left - self.seq_offset, 0)
        my_right = min(right - self.seq_offset, self.line_width - 1)

        half_x, left_rect_off, right_rect_off = self.base_layout_info()
        line_items = self.line_items[seq]
        item_aux_info = self.item_aux_info[seq]
        if self._large_alignment():
            res_status = hasattr(seq, "match_maps") and seq.match_maps
        else:
            res_status = seq in self.alignment.seqs
        color_func = self._color_func(seq)
        for i in range(my_left, my_right+1):
            line_item = line_items[i]
            if line_item is not None:
                line_item.hide()
                self.main_scene.removeItem(line_item)
            x, y = item_aux_info[i]
            line_items[i] = self.make_item(seq, self.seq_offset + i,
                        x, y, half_x, left_rect_off,
                        right_rect_off, color_func)
            if res_status:
                self._assoc_res_bind(line_items[i], seq, self.seq_offset + i)
        if self.show_numberings[0] and line_numbering_start(seq) is not None and my_left == 0:
            item = self.numbering_texts[seq][0]
            item.hide()
            self.main_scene.removeItem(item)
            self.numbering_texts[seq][0] = self._make_numbering(seq,0)
        if self.show_numberings[1] and line_numbering_start(seq) is not None \
                    and my_right == self.line_width - 1:
            item = self.numbering_texts[seq][1]
            item.hide()
            self.main_scene.removeItem(item)
            self.numbering_texts[seq][1] = self._make_numbering(seq,1)

    def relative_y(self, rawY):
        '''return the y relative to the block the y is in'''
        if rawY < self.top_y:
            if not self.prev_block:
                return 0
            else:
                return self.prev_block.relative_y(rawY)
        if rawY > self.bottom_y + self.block_gap:
            if not self.next_block:
                return self.bottom_y - self.top_y
            else:
                return self.next_block.relative_y(rawY)
        return min(rawY - self.top_y, self.bottom_y - self.top_y)

    def replace_label(self, line):
        self.label_texts[line].setText(line.name)
        if self.next_block:
            self.next_block.replace_label(line)

    def rerule(self):
        self.layout_ruler(rerule=True)
        if self.next_block:
            self.next_block.rerule()

    def row_index(self, y, bound=None):
        '''Given a relative y, return the row index'''
        rel_ruler_bottom = self.bottom_ruler_y - self.top_y
        if y <= rel_ruler_bottom:
            # in header
            if bound == "top":
                return 0
            elif bound == "bottom":
                if self.prev_block:
                    return len(self.lines) - 1
                return None
            return None
        row = int((y - rel_ruler_bottom) / (self.font_pixels[1] + self.letter_gaps[1]))
        if row >= len(self.lines):
            # off bottom
            if bound == "top":
                if self.next_block:
                    return 0
                return None
            elif bound == "bottom":
                return len(self.lines) - 1
            return None

        top_y = rel_ruler_bottom + row * (self.font_pixels[1] + self.letter_gaps[1])
        if y - top_y < self.letter_gaps[1]:
            # in gap
            if bound == "top":
                return row
            elif bound == "bottom":
                if row > 0:
                    return row - 1
                return len(self.lines) - 1
            return None
        # on letter
        return row

    def show_left_numbering(self, show_numbering):
        if show_numbering:
            delta = self.numbering_widths[0]
            for ruler_text in self.ruler_texts:
                ruler_text.moveBy(delta, 0)
            numbered_lines = [l for l in self.lines if line_numbering_start(l) is not None]
            for line in numbered_lines:
                self.numbering_texts[line][0] = self._make_numbering(line, 0)
            self._move_lines(self.lines, 0, delta, 0)
        else:
            delta = 0 - self.numbering_widths[0]
            for ruler_text in self.ruler_texts:
                ruler_text.moveBy(delta, 0)
            for texts in self.numbering_texts.values():
                if not texts[0]:
                    continue
                self.main_scene.removeItem(texts[0])
                texts[0] = None
            self._move_lines(self.lines, 0, delta, 0)
        if self.next_block:
            self.next_block.show_left_numbering(show_numbering)

    def show_right_numbering(self, show_numbering):
        if show_numbering:
            numbered_lines = [l for l in self.lines if line_numbering_start(l) is not None]
            for line in numbered_lines:
                self.numbering_texts[line][1] = self._make_numbering(line, 1)
        else:
            for texts in self.numbering_texts.values():
                if not texts[1]:
                    continue
                self.main_scene.removeItem(texts[1])
                texts[1] = None
        if self.next_block:
            self.next_block.show_right_numbering(show_numbering)

    def set_ruler_display(self, show_ruler, push_down=0):
        if show_ruler == self.show_ruler:
            return
        self.show_ruler = show_ruler
        self.top_y += push_down
        self.bottom_y += push_down
        pull = self.font_pixels[1] + self.letter_gaps[1]
        if show_ruler:
            push_down += pull
            self.layout_ruler()
        else:
            for text in self.ruler_texts:
                self.main_scene.removeItem(text)
            push_down -= pull
            self.bottom_ruler_y = self.top_y
            self.bottom_y -= pull
        self._move_lines(self.lines, 0, 0, push_down)
        self._move_tree(push_down)
        if self.next_block:
            self.next_block.set_ruler_display(show_ruler, push_down=push_down)

    def show_header(self, header, push_down=0):
        self.top_y += push_down
        if self.prev_block:
            self.label_width = self.prev_block.label_width
            insert_index = len(self.lines) - len(self.alignment.seqs) - 1
        else:
            insert_index = len(self.lines) - len(self.alignment.seqs)
            self.lines[insert_index:insert_index] = [header]
            for seq in self.alignment.seqs:
                self.line_index[seq] += 1
            self.line_index[header] = insert_index
            self.label_width = _find_label_width(self.lines, self.settings, self.font_metrics,
                self.emphasis_font_metrics, self.label_pad)

        for ruler_text in self.ruler_texts:
            ruler_text.moveBy(0, push_down)
        self.bottom_ruler_y += push_down

        self._move_lines(self.lines[:insert_index], 0, 0, push_down)

        self._layout_line(header, self.header_label_color, line_index=insert_index)
        push = self.font_pixels[1] + self.letter_gaps[1]
        push_down += push
        self._move_lines(self.lines[insert_index+1:], 0, 0, push_down)
        self._move_tree(push_down)
        self.bottom_y += push_down
        if self.next_block:
            self.next_block.show_header(header, push_down=push_down)

    """TODO
    def showNodes(self, show):
        if show:
            state = 'normal'
        else:
            state = 'hidden'
        for box in self.tree_items['boxes']:
            self.label_scene.itemconfigure(box, state=state)
        if self.next_block:
            self.next_block.showNodes(show)

    def showTree(self, treeInfo, callback, nodesShown, active=None):
        for box in self.tree_items['boxes']:
            self.tree_balloon.tagunbind(self.label_scene, box)
        for tree_items in self.tree_items.values():
            while tree_items:
                self.label_scene.delete(tree_items.pop())
        if self.next_block:
            self.next_block.showTree(treeInfo, callback, nodesShown)
        if not treeInfo:
            return
        self.treeNodeMap = {'active': active}
        self._layoutTree(treeInfo, treeInfo['tree'], callback,
                                nodesShown)

    def updateNumberings(self):
        numbered_lines = [l for l in self.lines if line_numberingStart(l) is not None]
        for line in numbered_lines:
            for i in range(2):
                nt = self.numbering_texts[line][i]
                if not nt:
                    continue
                self.main_scene.delete(nt)
                self.numbering_texts[line][i] = \
                        self._make_numbering(line, i)
        if self.next_block:
            self.next_block.updateNumberings()
"""

def _ellipsis_name(name, ellipsis_threshold):
    if len(name) > ellipsis_threshold:
        half = int(ellipsis_threshold/2)
        return name[0:half-1] + "..." + name[len(name)-half:]
    return name

def _find_label_width(lines, settings, font_metrics, emphasis_font_metrics, label_pad):
    label_width = 0
    for seq in lines:
        name = _seq_name(seq, settings)
        label_width = max(label_width, font_metrics.horizontalAdvance(name))
        label_width = max(label_width, emphasis_font_metrics.horizontalAdvance(name))
    label_width += label_pad
    return label_width

def _seq_name(seq, settings):
    """TODO
    return _ellipsis_name(seq.name, prefs[SEQ_NAME_ELLIPSIS])
    """
    return _ellipsis_name(seq.name, 30)

def _wrap_okay(num_seqs, settings):
    if num_seqs == 1:
        return getattr(settings, SINGLE_PREFIX + 'wrap')
    return num_seqs <= getattr(settings, ALIGNMENT_PREFIX + 'wrap_threshold')

def line_numbering_start(line):
    start = getattr(line, 'numbering_start', None)
    if start is None and isinstance(line, Sequence):
        start = 1
    return start
