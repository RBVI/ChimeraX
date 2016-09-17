# vim: set expandtab ts=4 sw=4:

"""TODO
from Consensus import Consensus
from Conservation import Conservation
from LineItem import LineItem
import string
from prefs import SINGLE_PREFIX
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
    """'public' methods are only public to the MultalignViewer class.
       Access to SeqCanvas functions is made through methods of the
       MultalignViewer class.
    """

    """TODO
	EditUpdateDelay = 7000
	viewMargin = 2
    """
	def __init__(self, parent, mav, alignment):
        from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QFrame, QHBoxLayout
        from PyQt5.QtCore import Qt
        self.label_scene = QGraphicsScene()
        self.label_scene.setBackgroundBrush(Qt.lightgray)
        self.label_view = QGraphicsView(self.label_scene)
        self._vdivider = QFrame()
        self._vdivider.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self._vdidver.setLineWidth(2)
        self._hdivider = QFrame()
        self._hdivider.setFrameStyle(QFrame.HLine | QFrame.Plain)
        self._hdivider.setLineWidth(1)
        self.main_scene = QGraphicsScene()
        self.main_scene.setBackgroundBrush(Qt.lightgray)
        self.main_view = QGraphicsView(self.main_scene)
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
		self.mav = mav
        self.alignment = alignment
        """TODO
		self.seqs = seqs
		for trig in [ADD_HEADERS, DEL_HEADERS,
				SHOW_HEADERS, HIDE_HEADERS, DISPLAY_TREE]:
			self.mav.triggers.addTrigger(trig)
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
		self.lineWidth = self.lineWidthFromPrefs()
		self.font = tkFont.Font(parent,
			(self.mav.prefs[FONT_NAME], self.mav.prefs[FONT_SIZE]))
        """
        from PyQt5.QtGui import QFont
        self.font = QFont("Helvetica")
        """TODO
		self.treeBalloon = Pmw.Balloon(parent)
		self.tree = self._treeCallback = None
		self.treeShown = self.nodesShown = False
		self._residueHandlers = None
        """
		self.layout_alignment()
        """TODO
		self.mainCanvas.grid(row=1, column=2, sticky='nsew')
		parent.columnconfigure(2, weight=1)
		parent.rowconfigure(1, weight=1)
        """
        layout = QHBoxLayout()
        layout.addWidget(self.label_view)
        layout.addWidget(self._hdivider)
        layout.addWidget(self.main_view, stretch=1)
        parent.setLayout(layout)
        self.main_view.show()
        """TODO

		# make the main canvas a reasonable size
		left, top, right, bottom = map(int,
				self.mainCanvas.cget("scrollregion").split())
		totalWidth = right - left + 1
		if self.shouldWrap():
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
		self._addDelSeqsHandler = self.mav.triggers.addHandler(
				ADDDEL_SEQS, self._addDelSeqsCB, None)
		self._seqRenamedHandler = self.mav.triggers.addHandler(
				SEQ_RENAMED, lambda *args: self._reformat(), None)
        """

    """TODO
	def activeNode(self):
		return self.leadBlock.treeNodeMap['active']

	def _addDelSeqsCB(self, trigName, myData, trigData):
		self._clustalXcache = {}
		for seq in self.seqs:
			try:
				cf = seq.colorFunc
			except AttributeError:
				continue
			break
		for seq in self.seqs:
			seq.colorFunc = cf
			if cf != self._cfBlack:
				self.recolor(seq)
		
	def addSeqs(self, seqs):
		for seq in seqs:
			self.labelBindings[seq] = {
				'<Enter>': lambda e, s=seq:
					self.mav.status(self.seqInfoText(s)),
				'<Double-Button>': lambda e, s=seq: self.mav._editSeqName(s)
			}
		self.mav.regionBrowser._preAddLines(seqs)
		self.leadBlock.addSeqs(seqs)
		self.mav.regionBrowser.redrawRegions()

	def adjustScrolling(self):
		self._resizescrollregion()
		self._recomputeScrollers()
		
	def _arrowCB(self, event):
		if event.state & 4 != 4:
			if event.keysym == "Up":
				self.mav.regionBrowser.raiseRegion(
						self.mav.currentRegion())
			elif event.keysym == "Down":
				self.mav.regionBrowser.lowerRegion(
						self.mav.currentRegion())
			else:
				self.mav.status(
					"Use control-arrow to edit alignment\n")
			return

		if event.keysym == "Up":
			self._undoRedo(False)
			return
		if event.keysym == "Down":
			self._undoRedo(True)
			return

		region = self.mav.currentRegion()
		if not region:
			replyobj.error("No active region.\n")
			return

		from RegionBrowser import SEL_REGION_NAME
		if region.name == SEL_REGION_NAME:
			replyobj.error(
				"Cannot edit using Chimera selection region\n")
			return

		if len(region.blocks) > 1:
			replyobj.error(
				"Cannot edit with multi-block region.\n")
			return

		line1, line2, pos1, pos2 = region.blocks[0]
		if line1 not in self.seqs:
			line1 = self.seqs[0]
		if line2 not in self.seqs:
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
		seqs = self.seqs[self.seqs.index(line1)
						:self.seqs.index(line2)+1]

		offset = 0
		if gapPos < 0 or gapPos >= len(line1):
			self.mav.status("Need to add columns to alignment to"
				" allow for requested motion.\nPlease wait...")
			# try to figure out the gap character
			# in use...
			gapChar = None
			for s in self.seqs:
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
							for x in self.seqs]
				start += num2add
				end += num2add
				pos1 += num2add
				pos2 += num2add
				gapPos += num2add
				offset = num2add
			else:
				newSeqs = [str(x) + gapChar * num2add
							for x in self.seqs]
			self.mav.realign(newSeqs, offset=offset,
							markEdited=True)
			self.mav.status("Columns added")
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
		self.mav._edited = True
		if incr == -1:
			left, right = gapPos, end-2-motion
		else:
			left, right = end+2-motion, gapPos
			
		self._editRefresh(seqs, left, right, region=region,
			lastBlock=[line1, line2, pos1+motion, pos2+motion])
		self._checkPoint(offset=offset, left=left, right=right)

	def addHeaders(self, headers):
		headers = [hd for hd in headers if hd not in self.headers]
		if not headers:
			return
		for hd in headers:
			self.labelBindings[hd] = {}
		self.headers.extend(headers)
		self.displayHeader.update({}.fromkeys(headers, False))
		self.mav.triggers.activateTrigger(ADD_HEADERS, headers)
		self.showHeaders(headers)

	def _associationsCB(self, trigName, myData, trigData):
		matchMaps = trigData[1]
		for mm in matchMaps:
			self.recolor(mm['aseq'])

	def assocSeq(self, aseq):
        """
		"""alignment sequence has gained or lost associated structure"""
        """
		self.leadBlock.assocSeq(aseq)

	def bboxList(self, line1, line2, pos1, pos2, coverGaps=True):
        """
		"""return coords that bound given lines and positions"""
        """
		return self.leadBlock.bboxList(line1, line2, pos1, pos2,
								coverGaps)

	def boundedBy(self, x1, y1, x2, y2):
        """
		"""return lines and offsets bounded by given coords"""
        """
		return self.leadBlock.boundedBy(x1, y1, x2, y2)

	def _attrsUpdateCB(self):
		self._delayedAttrsHandler = None
		self.mav.status("Updating residue attributes")
		self.mav.setResidueAttrs()
		self.mav.status("Residue attributes updated")

	def _checkPoint(self, fromScratch=False, checkChange=False, offset=0,
						left=None, right=None):
		if fromScratch:
			self._checkPoints = []
			self._checkPointIndex = -1
		self._checkPoints = self._checkPoints[:self._checkPointIndex+1]
		chkpt = [s[:] for s in self.seqs]
		if checkChange:
			if chkpt == self._checkPoints[self._checkPointIndex][0]:
				return
		self._checkPoints.append(
			(chkpt, (offset, self.mav._edited, left, right)))
		self._checkPointIndex += 1

	def _configureCB(self, e):
		# size change; scrollbars?
		import sys
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

	def _copyCB(self, e):
		region = self.mav.currentRegion()
		if region is None:
			copy = "\n".join([s.ungapped() for s in self.seqs])
		else:
			texts = {}
			for line1, line2, pos1, pos2 in region.blocks:
				try:
					i1 = self.seqs.index(line1)
				except ValueError:
					i1 = 0
				try:
					i2 = self.seqs.index(line2)
				except ValueError:
					continue
				for seq in self.seqs[i1:i2+1]:
					text = "".join([seq[p] for p in range(pos1, pos2+1)
								if seq.gapped2ungapped(p) is not None])
					if text:
						texts[seq] = texts.setdefault(seq, "") + text
			if not texts:
				self.mav.status("Active region is all gaps!", color="red")
				return
			copy = "\n".join([texts[seq] for seq in self.seqs
							if seq in texts])
		self.mainCanvas.clipboard_clear()
		self.mainCanvas.clipboard_append(copy)
		if region is None:
			if len(self.seqs) > 1:
				self.mav.status("No current region; copied all sequences")
			else:
				self.mav.status("Sequence copied")
		else:
			self.mav.status("Region copied")

	def dehighlightName(self):
		self.leadBlock.dehighlightName()

	def deleteHeaders(self, headers):
		if not headers:
			return
		for header in headers:
			if header in self.seqs:
				raise ValueError(
					"Cannot delete an alignment sequence")
			if header in self.builtinHeaders:
				raise ValueError("Cannot delete builtin header"
							" sequence")
		self.hideHeaders(headers)
		for hd in headers:
			del self.displayHeader[hd]
			self.headers.remove(hd)
		self.mav.triggers.activateTrigger(DEL_HEADERS, headers)

	def destroy(self):
		chimera.triggers.deleteHandler('Molecule', self._trigID)
		from MAViewer import ADDDEL_SEQS, SEQ_RENAMED
		self.mav.triggers.deleteHandler(ADDDEL_SEQS,
						self._addDelSeqsHandler)
		self.mav.triggers.deleteHandler(SEQ_RENAMED,
						self._seqRenamedHandler)
		if self._residueHandlers:
			chimera.triggers.deleteHandler('Residue',
						self._residueHandlers[0])
			from MAViewer import MOD_ASSOC
			self.mav.triggers.deleteHandler(MOD_ASSOC,
						self._residueHandlers[1])
		for header in self.headers:
			header.destroy()
		self.leadBlock.destroy()
		
	def _editHdrCB(self):
		left, right = self._editBounds
		self._editBounds = None
		for header in self.headers:
			if not hasattr(header, 'alignChange'):
				continue
			if header.fastUpdate():
				# already updated
				continue
			self.mav.status("Updating %s header" % header.name)
			header.alignChange(left, right)
			self.refresh(header, left=left, right=right)
			self.mav.status("%s header updated" % header.name)

	def _editRefresh(self, seqs, left, right, region=None, lastBlock=None):
		for header in self.headers:
			if not hasattr(header, 'alignChange'):
				continue
			if not self.displayHeader[header]:
				continue
			if not header.fastUpdate():
				# header can't update quickly; delay it
				self.mav.status("Postponing update of %s header"
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
		self.mav.regionBrowser.redrawRegions(justGapping=True)
		if not self._editBounds:
			if self._delayedAttrsHandler:
				self.mainCanvas.after_cancel(
						self._delayedAttrsHandler)
			self._delayedAttrsHandler = self.mainCanvas.after(
				self.EditUpdateDelay, self._attrsUpdateCB)
	def _escapeCB(self, event):
		if event.state & 4 != 4:
			self.mav.status(
				"Use control-escape to revert to unedited\n")
			return
		while self._checkPointIndex > 0:
			self._undoRedo(True)

	def headerDisplayOrder(self):
		return self.leadBlock.lines[:-len(self.seqs)]

	def hideHeaders(self, headers, fromMenu=False):
		headers = [hd for hd in headers if self.displayHeader[hd]]
		if not headers:
			return

		# only handle headers in continuous blocks...
		if len(headers) > 1:
			continuous = True
			li = self.leadBlock.lineIndex[headers[0]]
			for header in headers[1:]:
				li += 1
				if self.leadBlock.lineIndex[header] != li:
					continuous = False
					break
			if not continuous:
				jump = headers.index(header)
				self.hideHeaders(headers[:jump], fromMenu=fromMenu)
				self.hideHeaders(headers[jump:], fromMenu=fromMenu)
				return
		for header in headers:
			header.hide()
		if fromMenu and len(self.seqs) > 1:
			startHeaders = set(self.mav.prefs[STARTUP_HEADERS])
			startHeaders -= set([hd.name for hd in headers])
			self.mav.prefs[STARTUP_HEADERS] = startHeaders
		self.displayHeader.update({}.fromkeys(headers, False))
		self.mav.regionBrowser._preDelLines(headers)
		self.leadBlock.hideHeaders(headers)
		self.mav.regionBrowser.redrawRegions(cullEmpty=True)
		self.mav.triggers.activateTrigger(HIDE_HEADERS, headers)
        """
		
	def layout_alignment(self):
        """
		self.consensus = Consensus(self.mav)
		def colorConsensus(line, offset, upper=string.uppercase):
			if line[offset] in upper:
				if line.conserved[offset]:
					return 'red'
				return 'purple'
			return 'black'
		self.consensus.colorFunc = colorConsensus
		self.conservation = Conservation(self.mav, evalWhileHidden=True)
		self.headers = [self.consensus, self.conservation]
		self.builtinHeaders = self.headers[:]
		startupHeaders = self.mav.prefs[STARTUP_HEADERS]
		useDispDefault = startupHeaders == None
		if useDispDefault:
			startupHeaders = set([self.consensus.name, self.conservation.name])
		from HeaderSequence import registeredHeaders, \
					DynamicStructureHeaderSequence
		for seq, defaultOn in registeredHeaders.values():
			header = seq(self.mav)
			self.headers.append(header)
			if useDispDefault and defaultOn:
				startupHeaders.add(header.name)
		if useDispDefault:
			self.mav.prefs[STARTUP_HEADERS] = startupHeaders
		singleSequence = len(self.seqs) == 1
		self.displayHeader = {}
		for header in self.headers:
			show = self.displayHeader[header] = header.name in startupHeaders \
				and not (singleSequence and not header.singleSequenceRelevant) \
				and not isinstance(header, DynamicStructureHeaderSequence)
			if show:
				header.show()
		self.headers.sort(lambda s1, s2: cmp(s1.sortVal, s2.sortVal)
						or cmp(s1.name, s2.name))
		self.labelBindings = {}
		for seq in self.seqs:
			self.labelBindings[seq] = {
				'<Enter>': lambda e, s=seq:
					self.mav.status(self.seqInfoText(s)),
				'<Double-Button>': lambda e, s=seq: self.mav._editSeqName(s)
			}
		initialHeaders = [hd for hd in self.headers
						if self.displayHeader[hd]]
		for line in self.headers:
			self.labelBindings[line] = {}

		# first, set residue coloring to a known safe value...
		from clustalX import clustalInfo
		if singleSequence:
			prefResColor = RC_BLACK
		else:
			prefResColor = self.mav.prefs[RESIDUE_COLORING]
		if prefResColor == RC_BLACK:
			rc = self._cfBlack
		elif prefResColor == RC_RIBBON:
			from MAViewer import MOD_ASSOC
			self._residueHandlers = [chimera.triggers.addHandler(
					'Residue', self._resChangeCB, None),
				self.mav.triggers.addHandler(MOD_ASSOC,
					self._associationsCB, None)]
			rc = self._cfRibbon
		else:
			rc = self._cfClustalX
		for seq in self.seqs:
			seq.colorFunc = rc
		self._clustalXcache = {}
		self._clustalCategories, self._clustalColorings = clustalInfo()
		# try to set to external color scheme
		if prefResColor not in nonFileResidueColorings:
			try:
				self._clustalCategories, self._clustalColorings\
						= clustalInfo(prefResColor)
			except:
				schemes = self.mav.prefs[RC_CUSTOM_SCHEMES]
				if prefResColor in schemes:
					schemes.remove(prefResColor)
					self.mav.prefs[
						RC_CUSTOM_SCHEMES] = schemes[:]
				self.mav.prefs[RESIDUE_COLORING] = RC_CLUSTALX
				from sys import exc_info
				replyobj.error("Error reading %s: %s\nUsing"
					" default ClustalX coloring instead\n"
					% (prefResColor, exc_info()[1]))

		self.showRuler = self.mav.prefs[SHOW_RULER_AT_STARTUP] \
						and not singleSequence
		self.showNumberings = [self.mav.leftNumberingVar.get(),
					self.mav.rightNumberingVar.get()]
		self.leadBlock = SeqBlock(self._labelCanvas(), self.mainCanvas,
			None, self.font, 0, initialHeaders, self.seqs,
			self.lineWidth, self.labelBindings, lambda *args, **kw:
			self.mav.status(secondary=True, *args, **kw),
			self.showRuler, self.treeBalloon, self.showNumberings,
			self.mav.prefs)
		self._resizescrollregion()
        """
        self.lead_block = SeqBlock(self._label_scene(), self.main_scene,
            None, self.font, 0 [], self.alignment,
            50, {}, lambda *args, **kw: self.mav.status(secondary=True, *args, **kw),
            False, None, False, None)

    """TODO
	def lineWidthFromPrefs(self):
		if self.shouldWrap():
			if len(self.mav.seqs) == 1:
				prefix = SINGLE_PREFIX
			else:
				prefix = ""
			return self.mav.prefs[prefix + LINE_WIDTH]
		# lay out entire sequence horizontally
		return 2 * len(self.seqs[0])

	def _molChange(self, trigger, myData, changes):
		# molecule attributes changed

		# find sequences (if any) with associations to changed mols
		assocSeqs = []
		for mol in changes.created | changes.modified:
			try:
				seq = self.mav.associations[mol]
			except KeyError:
				continue
			if seq not in assocSeqs:
				assocSeqs.append(seq)
		if assocSeqs:
			self.leadBlock._molChange(assocSeqs)

	def _multiScroll(self, *args):
		self.labelCanvas.yview(*args)
		self.mainCanvas.yview(*args)

	def _newFont(self):
		if len(self.mav.seqs) == 1:
			prefix = SINGLE_PREFIX
		else:
			prefix = ""
		fontname, fontsize = (self.mav.prefs[prefix + FONT_NAME],
						self.mav.prefs[prefix + FONT_SIZE])
		self.font = tkFont.Font(self.mainCanvas, (fontname, fontsize))
		self.mav.status("Changing to %d point %s"
					% (fontsize, fontname), blankAfter=0)
		self.leadBlock.fontChange(self.font)
		self.refreshTree()
		self.mav.regionBrowser.redrawRegions()
		self.mav.status("Font changed")

	def _newWrap(self):
        """
		"""alignment wrapping preferences have changed"""
        """
		lineWidth = self.lineWidthFromPrefs()
		if lineWidth == self.lineWidth:
			return
		self.lineWidth = lineWidth
		self._reformat()

	def _pageDownCB(self, event):
		numBlocks = self.leadBlock.numBlocks()
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
		numBlocks = self.leadBlock.numBlocks()
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
		rb = self.mav.regionBrowser
		if handleRegions:
			# do what we can; move single-seq regions that begin and end
			# over non-gap characters, delete others
			deleteRegions = []
			regionUpdateInfo = []
			for region in rb.regions:
				if not region.blocks:
					continue
				seq = region.blocks[0][0]
				if seq not in self.mav.seqs:
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
		prevLen = len(self.seqs[0])
		for i in range(len(seqs)):
			self.seqs[i][:] = seqs[i]
		for header in self.headers:
			header.reevaluate()
		self._clustalXcache = {}
		self.leadBlock.realign(prevLen)
		if savedSNs[0]:
			self.setLeftNumberingDisplay(True)
		if savedSNs[1]:
			self.setRightNumberingDisplay(True)
		self._resizescrollregion()
		if len(self.seqs[0]) != prevLen:
			self._recomputeScrollers()
		if handleRegions:
			for region, seq, ungappedBlocks in regionUpdateInfo:
				blocks = []
				for ub in ungappedBlocks:
					gb = [seq.ungapped2gapped(x) for x in ub]
					blocks.append([seq, seq] + gb)
				region.addBlocks(blocks)

	def recolor(self, seq):
		self.leadBlock.recolor(seq)

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

		if not self.shouldWrap() and self._vscrollMapped:
			self._hdivider.grid(row=2, column=0, sticky="new")
		else:
			self._hdivider.grid_forget()

	def _reformat(self, cullEmpty=False):
		self.mav.status("Reformatting alignment; please wait...",
								blankAfter=0)
		if self.tree:
			activeNode = self.activeNode()
		self.leadBlock.destroy()
		initialHeaders = [hd for hd in self.headers
						if self.displayHeader[hd]]
		self.leadBlock = SeqBlock(self._labelCanvas(), self.mainCanvas,
			None, self.font, 0, initialHeaders, self.seqs,
			self.lineWidth, self.labelBindings, lambda *args, **kw:
			self.mav.status(secondary=True, *args, **kw),
			self.showRuler, self.treeBalloon, self.showNumberings,
			self.mav.prefs)
		if self.tree:
			if self.treeShown:
				self.leadBlock.showTree({'tree': self.tree},
					self._treeCallback, self.nodesShown,
					active=activeNode)
			else:
				self.leadBlock.treeNodeMap = {'active':
								activeNode }
		self.mav.regionBrowser.redrawRegions(cullEmpty=cullEmpty)
		if len(self.seqs) != len(self._checkPoints[0]):
			self._checkPoint(fromScratch=True)
		else:
			self._checkPoint(checkChange=True)
		self.mav.status("Alignment reformatted")

	def refresh(self, seq, left=0, right=None, updateAttrs=True):
		if seq in self.displayHeader and not self.displayHeader[seq]:
			return
		if right is None:
			right = len(self.seqs[0])-1
		self.leadBlock.refresh(seq, left, right)
		if updateAttrs:
			self.mav.setResidueAttrs()

	def refreshTree(self):
		if self.treeShown:
			self.leadBlock.showTree({'tree': self.tree},
					self._treeCallback, self.nodesShown,
					active=self.activeNode())

	def _resChangeCB(self, trigName, myData, trigData):
		mols = set([r.molecule for r in trigData.modified])
		for m in mols:
			if m in self.mav.associations:
				self.recolor(self.mav.associations[m])

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
		self.mav.status(msg)

	def seeBlocks(self, blocks):
        """
		"""scroll canvas to show given blocks"""
        """
		minx, miny, maxx, maxy = self.bboxList(coverGaps=True,
								*blocks[0])[0]
		for block in blocks:
			for x1, y1, x2, y2 in self.bboxList(coverGaps=True,
								*block):
				minx = min(minx, x1)
				miny = min(miny, y1)
				maxx = max(maxx, x2)
				maxy = max(maxy, y2)
		viewWidth = float(self.mainCanvas.cget('width'))
		viewHeight = float(self.mainCanvas.cget('height'))
		if maxx - minx > viewWidth or maxy - miny > viewHeight:
			# blocks don't fit in view; just show first block
			minx, miny, maxx, maxy = self.bboxList(coverGaps=True,
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
		if not self.shouldWrap():
			self.labelCanvas.yview_moveto(starty)
		self.mainCanvas.yview_moveto(starty)

	def seeSeq(self, seq, highlightName):
        """
		"""scroll up/down to center given seq, and possibly highlight name"""
        """
		minx, miny, maxx, maxy = self.bboxList(seq, seq, 0, 0)[0]
		viewHeight = float(self.mainCanvas.cget('height'))
		cy = (miny + maxy) / 2
		
		x1, y1, x2, y2 = map(int,
			self.mainCanvas.cget('scrollregion').split())
		totalHeight = float(y2 - y1 + 1)

		if cy < y1 + viewHeight/2:
			cy = y1 + viewHeight/2
		starty = max(0.0, min((cy - viewHeight/2 - y1) / totalHeight,
					(y2 - viewHeight - y1) / totalHeight))
		if not self.shouldWrap():
			self.labelCanvas.yview_moveto(starty)
		self.mainCanvas.yview_moveto(starty)
		if highlightName:
			self.leadBlock.highlightName(seq)

	def setClustalParams(self, categories, colorings):
		self._clustalXcache = {}
		self._clustalCategories, self._clustalColorings = \
						categories, colorings
		if self.mav.prefs[RESIDUE_COLORING] in [RC_BLACK, RC_RIBBON]:
			return
		for seq in self.seqs:
			self.refresh(seq)

	def seqInfoText(self, aseq):
		basicText = "%s (#%d of %d; %d non-gap residues)\n" % (aseq.name,
			self.mav.seqs.index(aseq)+1, len(self.mav.seqs),
			len(aseq.ungapped()))
		if self.mav.intrinsicStructure \
				or not hasattr(aseq, 'matchMaps') or not aseq.matchMaps:
			return basicText
		return "%s%s associated with %s\n" % (basicText,
			seq_name(aseq, self.mav.prefs),
			", ".join(["%s (%s %s)" % (m.oslIdent(), m.name,
			aseq.matchMaps[m]['mseq'].name)
			for m in aseq.matchMaps.keys()]))

	def setColorFunc(self, coloring):
		if self._residueHandlers:
			chimera.triggers.deleteHandler('Residue',
						self._residueHandlers[0])
			from MAViewer import MOD_ASSOC
			self.mav.triggers.deleteHandler(MOD_ASSOC,
						self._residueHandlers[1])
			self._residueHandlers = None
		if coloring == RC_BLACK:
			cf = self._cfBlack
		elif coloring == RC_RIBBON:
			from MAViewer import MOD_ASSOC
			self._residueHandlers = [chimera.triggers.addHandler(
					'Residue', self._resChangeCB, None),
				self.mav.triggers.addHandler(MOD_ASSOC,
					self._associationsCB, None)]
			cf = self._cfRibbon
		else:
			cf = self._cfClustalX
		for seq in self.seqs:
			if not hasattr(seq, 'colorFunc'):
				seq.colorFunc = None
			if seq.colorFunc != cf:
				seq.colorFunc = cf
				self.recolor(seq)

	def setLeftNumberingDisplay(self, showNumbering):
		if self.showNumberings[0] == showNumbering:
			return
		self.showNumberings[0] = showNumbering
		self.leadBlock.setLeftNumberingDisplay(showNumbering)
		self.mav.regionBrowser.redrawRegions()
		self._resizescrollregion()
		self._recomputeScrollers()

	def setRightNumberingDisplay(self, showNumbering):
		if self.showNumberings[1] == showNumbering:
			return
		self.showNumberings[1] = showNumbering
		self.leadBlock.setRightNumberingDisplay(showNumbering)
		self._resizescrollregion()
		if showNumbering:
			self._recomputeScrollers(xShowAt=1.0)
		else:
			self._recomputeScrollers()

	def setRulerDisplay(self, showRuler):
		if showRuler == self.showRuler:
			return
		self.showRuler = showRuler
		self.leadBlock.setRulerDisplay(showRuler)
		self.mav.regionBrowser.redrawRegions(cullEmpty=not showRuler)

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
		for seq in self.seqs:
			char = seq[offset].lower()
			chars[char] = chars.get(char, 0) + 1
		consensusChars = {}
		numSeqs = float(len(self.seqs))

		for members, threshold, result in self._clustalCategories:
			sum = 0
			for c in members:
				sum += chars.get(c, 0)
			if sum / numSeqs >= threshold:
				consensusChars[result] = True

		self._clustalXcache[offset] = consensusChars
		return consensusChars
        """

	def should_wrap(self):
        return True
        """TODO
		return shouldWrap(len(self.seqs), self.mav.prefs)

	def showHeaders(self, headers, fromMenu=False):
		headers = [hd for hd in headers if not self.displayHeader[hd]]
		if not headers:
			return
		for header in headers:
			header.show()
		if fromMenu and len(self.seqs) > 1:
			startHeaders = set(self.mav.prefs[STARTUP_HEADERS])
			startHeaders |= set([hd.name for hd in headers])
			self.mav.prefs[STARTUP_HEADERS] = startHeaders
		self.displayHeader.update({}.fromkeys(headers, True))
		self.mav.regionBrowser._preAddLines(headers)
		self.leadBlock.showHeaders(headers)
		self.mav.regionBrowser.redrawRegions()
		self.mav.setResidueAttrs()
		self.mav.triggers.activateTrigger(SHOW_HEADERS, headers)

	def showNodes(self, show):
		if show == self.nodesShown:
			return
		self.nodesShown = show
		self.leadBlock.showNodes(show)

	def showTree(self, show):
		if show == self.treeShown or not self.tree:
			return

		if show:
			self.leadBlock.showTree({'tree': self.tree},
						self._treeCallback, True,
						active=self.activeNode())
			self.mav.triggers.activateTrigger(
							DISPLAY_TREE, self.tree)
		else:
			self.leadBlock.showTree(None, None, None)
			self.mav.triggers.activateTrigger(DISPLAY_TREE, None)
		self._resizescrollregion()
		self._recomputeScrollers(xShowAt=0.0)
		self.treeShown = show

	def updateNumberings(self):
		self.leadBlock.updateNumberings()
		self._resizescrollregion()

	def usePhyloTree(self, tree, callback=None):
		treeInfo = {}
		if tree:
			tree.assignYpositions()
			tree.assignXpositions(branchStyle="weighted")
			tree.assignXdeltas()
			treeInfo['tree'] = tree
		self.leadBlock.showTree(treeInfo, callback, True)
		self.leadBlock.activateNode(tree)
		self._resizescrollregion()
		self._recomputeScrollers(xShowAt=0.0)
		self.tree = tree
		self._treeCallback = callback
		self.treeShown = self.nodesShown = bool(tree)
		self.mav.triggers.activateTrigger(DISPLAY_TREE, tree)
        """

	def _label_scene(self, grid=true):
		if self.should_wrap():
			label_scene = self.main_scene
			if grid:
				self.label_view.hide()
				self._vdivider.hide()
		else:
			label_scene = self.label_scene
			if grid:
				self.label_view.show()
				self._vdivider.show()
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
		self.mav._edited = chkEdited
		if len(checkPoint[0]) != len(self.seqs[0]):
			self.mav.status("Need to change number of columns in"
				" alignment to allow for requested change.\n"
				"Please wait...")
			self.mav.realign(checkPoint, offset=offset)
			self.mav.status("Columns changed")
			return
		for seq, chkSeq in zip(self.seqs, checkPoint):
			seq[:] = chkSeq
		self._editRefresh(self.seqs, left, right)
        """


class SeqBlock:
    from PyQt5.QtCore import Qt
	normal_label_color = Qt::black
	header_label_color = Qt::blue

	def __init__(self, label_scene, main_scene, prev_block, font, seq_offset,
			headers, alignment, line_width, label_bindings, status_func,
			show_ruler, tree_balloon, show_numberings, prefs):
		self.label_scene = label_scene
		self.main_scene = main_scene
		self.prev_block = prev_block
		self.alignment = alignment
		self.font = font
		self.label_bindings = label_bindings
		self.status_func = status_func
        """TODO
		self._mouseID = None
		if len(seqs) == 1:
			prefPrefix = SINGLE_PREFIX
		else:
			prefPrefix = ""
        """
        self.letter_gaps = [0, 1]
        """TODO
		self.letter_gaps = [prefs[prefPrefix + COLUMN_SEP],
						prefs[prefPrefix + LINE_SEP]]
		if prefs[prefPrefix + TEN_RES_GAP]:
			self.chunkGap = 20
		else:
			self.chunkGap = 0
        """
        self.chunkGap = 0
        """TODO
		if prefs[prefPrefix + BLOCK_SPACE]:
			self.blockGap = 15
		else:
			self.blockGap = 0
		self.show_ruler = show_ruler
		self.tree_balloon = tree_balloon
		self.show_numberings = show_numberings[:]
        """
		self.prefs = prefs
		self.seq_offset = seq_offset
		self.line_width = line_width

		if prev_block:
			self.top_y = prev_block.bottomY + self.blockGap
			self.label_width = prev_block.label_width
			self.font_pixels = prev_block.font_pixels
			self.lines = prev_block.lines
			self.line_index = prev_block.line_index
			self.emphasis_font = prev_block.emphasis_font
			self.emphasis_font_metrics = prev_block.emphasis_font_metrics
			self.font_metrics = prev_block.font_metrics
			self.numbering_widths = prev_block.numbering_widths
            self._brushes = prev_block._brushes
		else:
			self.top_y = 0
			self.line_index = {}
			lines = list(headers) + list(self.alignment.seqs)
			for i in range(len(lines)):
				self.line_index[lines[i]] = i
			self.lines = lines
            from PyQt5.QtGui import QFont, QFontMetrics
			self.emphasis_font = QFont(self.font)
			self.emphasis_font.setBond(True)
            """TODO
			if prefs[prefPrefix + BOLD_ALIGNMENT]:
				self.font = self.emphasis_font
            """
            self.font_metrics = QFontMetrics(self.font)
            self.emphasis_font_metrics = QFontMetrics(self.emphasis_font)
			self.label_width = self.find_label_width(self.font_metrics, self.emphasis_font_metrics)
			font_width, font_height = self.font_metrics.maxWidth(), self.font_metrics.height()
			# pad font a little...
			self.font_pixels = (font_width + 1, font_height + 1)
            self._brushes = {}
            """TODO
			self.numbering_widths = self.findNumberingWidths(self.font)
            """
			# long sequences can cause deep recursion...
			import sys
			recur_limit = sys.getrecursionlimit()
			# seems to be a hidden factor of 4 between the recursion
			# limit and the actual stack depth (!)
			if 4 * (100 + len(seqs[0]) / line_width) > recur_limit:
				sys.setrecursionlimit(4 * int(100 + len(seqs[0]) / line_width))
		self.bottom_y = self.top_y

		self.label_texts = {}
        """TODO
		self.labelRects = {}
		self.numberingTexts = {}
		self.lineItems = {}
		self.itemAuxInfo = {}
		self.treeItems = { 'lines': [], 'boxes': [] }
		self.highlightedName = None
        """

		self.layout_ruler()
		self.layout_lines(headers, self.header_label_color)
		self.layout_lines(alignment.seqs, self.normal_label_color)

		if seq_offset + line_width >= len(alignment.seqs[0]):
			self.next_block = None
		else:
			self.next_block = SeqBlock(label_scene, main_scene,
				self, self.font, seq_offset + line_width, headers,
				alignment, line_width, label_bindings, status_func,
				show_ruler, tree_balloon, show_numberings,
				self.prefs)

    """
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
			newNumberingWidths = self.findNumberingWidths(self.font)
		labelChange = newLabelWidth - self.label_width
		self.label_width = newLabelWidth
		numberingChanges = [newNumberingWidths[i]
				- self.numbering_widths[i] for i in range(2)]
		self.numbering_widths = newNumberingWidths

		for rulerText in self.ruler_texts:
			self.main_scene.move(rulerText,
				labelChange + numberingChanges[0], pushDown)
		self.bottom_ruler_y += pushDown

		self._moveLines(self.lines[:insertIndex], labelChange,
						numberingChanges[0], pushDown)

		for i, seq in enumerate(seqs):
			self._layout_line(seq, self.normal_label_color,
						line_index=insertIndex+i, adding=True)
		push = len(seqs) * (self.font_pixels[1] + self.letter_gaps[1])
		pushDown += push
		self._moveLines(self.lines[insertIndex+len(seqs):],
				labelChange, numberingChanges[0], pushDown)
		self.bottom_y += pushDown
		if self.next_block:
			self.next_block.addSeqs(seqs, pushDown=pushDown)

	def _assocResBind(self, item, aseq, index):
		item.tagBind('<Enter>', lambda e: self._mouseResidue(1, aseq, index))
		item.tagBind('<Leave>', lambda e: self._mouseResidue(0))

	def assocSeq(self, aseq):
		item = self.label_texts[aseq]
		self.label_scene.itemconfigure(item, font=self._label_font(aseq))
		associated = self.has_associated_structures(aseq)
		if associated:
			self._colorizeLabel(aseq)
		else:
			if self.labelRects.has_key(aseq):
				self.label_scene.delete(self.labelRects[aseq])
				del self.labelRects[aseq]
		if self._largeAlignment():
			lineItems = self.lineItems[aseq]
			for i in range(len(lineItems)):
				item = lineItems[i]
				if not item:
					continue
				if associated:
					self._assocResBind(item, aseq, self.seq_offset+i)
				else:
					item.tagBind('<Enter>', "")
					item.tagBind('<Leave>', "")
		if self.next_block:
			self.next_block.assocSeq(aseq)
    """
		
	def base_layout_info(self):
		half_x = self.font_pixels[0] / 2
		left_rect_off = 0 - half_x
		right_rect_off = self.font_pixels[0] - half_x
		return half_x, left_rect_off, right_rect_off

    """TODO
	def bboxList(self, line1, line2, pos1, pos2, coverGaps):
		if pos1 >= self.seq_offset + self.line_width:
			return self.next_block.bboxList(line1, line2, pos1, pos2,
								coverGaps)
		left = max(pos1, self.seq_offset) - self.seq_offset
		right = min(pos2, self.seq_offset + self.line_width - 1) \
							- self.seq_offset
		bboxes = []
		if coverGaps:
			bboxes.append(self._boxCorners(left,right,line1,line2))
		else:
			l1 = self.line_index[line1]
			l2 = self.line_index[line2]
			lmin = min(l1, l2)
			lmax = max(l1, l2)
			for line in self.lines[l1:l2+1]:
				l = None
				for lo in range(left, right+1):
					if line.gapped2ungapped(lo +
							self.seq_offset) is None:
						# gap
						if l is not None:
							bboxes.append(self._boxCorners(l, lo-1, line, line))
							l = None
					else:
						# not gap
						if l is None:
							l = lo
				if l is not None:
					bboxes.append(self._boxCorners(
							l, right, line, line))
							
		if pos2 >= self.seq_offset + self.line_width:
			bboxes.extend(self.next_block.bboxList(
					line1, line2, pos1, pos2, coverGaps))
		return bboxes

	def _boxCorners(self, left, right, line1, line2):
		ulx = self._left_seqs_edge() + left * (
				self.letter_gaps[0] + self.font_pixels[0]) \
				+ int(left/10) * self.chunkGap
		uly = self.bottom_ruler_y + self.letter_gaps[1] \
				+ self.line_index[line1] * (
				self.font_pixels[1] + self.letter_gaps[1])
		lrx = self._left_seqs_edge() - self.letter_gaps[0] + (right+1) * (
				self.letter_gaps[0] + self.font_pixels[0]) \
				+ int(right/10) * self.chunkGap
		lry = self.bottom_ruler_y + (self.line_index[line2] + 1) * (
				self.font_pixels[1] + self.letter_gaps[1])
		if len(self.seqs) == 1:
			prefPrefix = SINGLE_PREFIX
		else:
			prefPrefix = ""
		if self.prefs[prefPrefix + COLUMN_SEP] < -1:
			overlap = int(abs(self.prefs[prefPrefix + COLUMN_SEP]) / 2)
			ulx += overlap
			lrx -= overlap
		return ulx, uly, lrx, lry

	def boundedBy(self, x1, y1, x2, y2):
		end = self.bottom_y + self.blockGap
		if y1 > end and y2 > end:
			if self.next_block:
				return self.next_block.boundedBy(x1, y1, x2, y2)
			else:
				return (None, None, None, None)
		relY1 = self.relativeY(y1)
		relY2 = self.relativeY(y2)
		if relY1 < relY2:
			hiRow = self.rowIndex(relY1, bound="top")
			lowRow = self.rowIndex(relY2, bound="bottom")
		else:
			hiRow = self.rowIndex(relY2, bound="top")
			lowRow = self.rowIndex(relY1, bound="bottom")
		if hiRow is None or lowRow is None:
			return (None, None, None, None)

		if y1 <= end and y2 <= end:
			if y1 > self.bottom_y and y2 > self.bottom_y \
			or y1 <= self.bottom_ruler_y and y2 <= self.bottom_ruler_y:
				# entirely in the same block gap or ruler
				return (None, None, None, None)
			# both on this block; determine right and left...
			leftX = min(x1, x2)
			rightX = max(x1, x2)
			leftPos = self.pos(leftX, bound="left")
			rightPos = self.pos(rightX, bound="right")
		else:
			# the one on this block is left...
			if y1 <= end:
				leftX, rightX, lowY = x1, x2, y2
			else:
				leftX, rightX, lowY = x2, x1, y1
			leftPos = self.pos(leftX, bound="left")
			if self.next_block:
				rightPos = self.next_block.pos(rightX,
							bound="right", y=lowY)
			else:
				rightPos = self.pos(rightX, bound="right")
		if leftPos is None or rightPos is None or leftPos > rightPos:
			return (None, None, None, None)
		return (self.lines[hiRow], self.lines[lowRow],
							leftPos, rightPos)

    """

    def _brush(self, color, item):
        try:
            return self._brushes[color]
        except KeyError:
            brush = item.brush()
            brush.setColor(color)
            self._brushes[color] = brush
            return brush

    """TODO
	def _colorFunc(self, line):
		try:
			return line.colorFunc
		except AttributeError:
			return lambda l, o: 'black'

	def _colorizeLabel(self, aseq):
		labelText = self.label_texts[aseq]
		bbox = self.label_scene.bbox(labelText)
		if self.labelRects.has_key(aseq):
			labelRect = self.labelRects[aseq]
			self.label_scene.coords(labelRect, *bbox)
		else:
			labelRect = self.label_scene.create_rectangle(*bbox)
			self.label_scene.tag_lower(labelRect, labelText)
			self.labelRects[aseq] = labelRect
		if len(aseq.matchMaps) > 1:
			stipple = ""
			color = ""
			dash = "."
			outline = "dark green"
		else:
			from CGLtk.color import rgba2tk
			color = rgba2tk(aseq.matchMaps.keys()[0].color.rgba())
			stipple= ""
			dash = ""
			outline = ""
		self.label_scene.itemconfigure(labelRect, stipple=stipple,
					dash=dash, outline=outline, fill=color)

	def _computeNumbering(self, line, end):
		if end == 0:
			count = len([c for c in line[:self.seq_offset]
						if c.isalpha()])
			if count == len(line.ungapped()):
				count -= 1
		else:
			count = len([c for c in line[:self.seq_offset
				+ self.line_width] if c.isalpha()]) - 1
		return line.numberingStart + count

	def dehighlightName(self):
		if self.highlightedName:
			self.label_scene.itemconfigure(self.highlightedName,
				fill=self.normal_label_color)
			self.highlightedName = None
			if self.next_block:
				self.next_block.dehighlightName()

	def destroy(self):
		if self.next_block:
			self.next_block.destroy()
			self.next_block = None
		for rulerText in self.ruler_texts:
			self.main_scene.delete(rulerText)
		for labelText in self.label_texts.values():
			self.label_scene.delete(labelText)
		for numberings in self.numberingTexts.values():
			for numbering in numberings:
				if numbering:
					self.main_scene.delete(numbering)
		for box in self.treeItems['boxes']:
			self.tree_balloon.tagunbind(self.label_scene, box)
		for treeItems in self.treeItems.values():
			for treeItem in treeItems:
				self.label_scene.delete(treeItem)
		for labelRect in self.labelRects.values():
			self.label_scene.delete(labelRect)
		for lineItems in self.lineItems.values():
			for lineItem in lineItems:
				if lineItem is not None:
					lineItem.delete()
    """
		
	def find_label_width(self, font_metrics, emphasis_font_metrics):
		label_width = 0
		for seq in self.lines:
			name = seq_name(seq, self.prefs)
			label_width = max(label_width, font_metrics.width(name))
			label_width = max(label_width, emphasis_font_metrics.width(name))
		label_width += 3
		return label_width

    """TODO
	def findNumberingWidths(self, font):
		lwidth = rwidth = 0
		if self.show_numberings[0]:
			baseNumBlocks = int(len(self.seqs[0]) / self.line_width)
			blocks = baseNumBlocks + (baseNumBlocks !=
					len(self.seqs[0]) / self.line_width)
			extent = (blocks - 1) * self.line_width
			for seq in self.lines:
				if getattr(seq, 'numberingStart', None) == None:
					continue
				offset = len([c for c in seq[:extent]
							if c.isalpha()])
				lwidth = max(lwidth, font.measure(
					"%d " % (seq.numberingStart + offset)))
			lwidth += 3
		if self.show_numberings[1]:
			for seq in self.lines:
				if getattr(seq, 'numberingStart', None) == None:
					continue
				offset = len(seq.ungapped())
				rwidth = max(rwidth, font.measure(
					"  %d" % (seq.numberingStart + offset)))
		return [lwidth, rwidth]

	def fontChange(self, font, emphasis_font=None, pushDown=0):
		self.top_y += pushDown
		self.font = font
		if emphasis_font:
			self.emphasis_font = emphasis_font
		else:
			self.emphasis_font = self.font.copy()
			self.emphasis_font.configure(weight=tkFont.BOLD)
		if len(self.seqs) == 1:
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
			newNumberingWidths = self.findNumberingWidths(font)
		labelChange = newLabelWidth - self.label_width
		leftNumberingChange = newNumberingWidths[0] \
						- self.numbering_widths[0]
		curWidth, curHeight = self.font_pixels
		newWidth, newHeight = font_pixels

		perLine = newHeight - curHeight
		perChar = newWidth - curWidth

		over = labelChange + leftNumberingChange + perChar / 2
		down = pushDown + perLine
		for rulerText in self.ruler_texts:
			self.main_scene.itemconfigure(rulerText, font=font)
			self.main_scene.move(rulerText, over, down)
			over += perChar * 10
		self.bottom_ruler_y += down

		down = pushDown + 2 * perLine
		for line in self.lines:
			labelText = self.label_texts[line]
			self.label_scene.itemconfigure(labelText,
						font=self._label_font(line))
			self.label_scene.move(labelText, 0, down)
			if self.labelRects.has_key(line):
				self._colorizeLabel(line)
			leftNumberingText = self.numberingTexts[line][0]
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
			lineItems = self.lineItems[seq]
			itemAuxInfo = self.itemAuxInfo[seq]
			colorFunc = self._colorFunc(seq)
			depictable = hasattr(seq, 'depictionVal')
			for i in range(len(lineItems)):
				lineItem = lineItems[i]
				oldx, oldy = itemAuxInfo[i]
				newx, newy = itemAuxInfo[i] = (oldx + over, oldy + down)
				if lineItem:
					index = self.seq_offset + i
					if depictable:
						val = seq.depictionVal(index)
					else:
						val = seq[index]
					if isinstance(val, basestring):
						if len(val) > 1:
							lineItem.delete()
							left = newx - newWidth/2.0
							right = newx + newWidth/2.0
							top = newy - newHeight
							lineItem.draw(val, left, right, top, newy,
											colorFunc(seq, index))
						else:
							lineItem.move(over, down)
							lineItem.configure(font=self.font)
					else:
						leftX, top_y, rightX, bottom_y = lineItem.coords()
						leftX += over - histLeft
						rightX += over + histRight
						oldHeight = bottom_y - top_y
						bottom_y += down
						top_y = bottom_y - (newHeight * float(oldHeight)
								/ curHeight)
						lineItem.coords(leftX, top_y, rightX, bottom_y)

				over += perChar
			rightNumberingText = self.numberingTexts[seq][1]
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

	def _getXs(self, amount):
		xs = []
		halfX, leftRectOff, rightRectOff = self.base_layout_info()
		x = self._left_seqs_edge() + halfX
		for chunkStart in range(0, amount, 10):
			for offset in range(chunkStart,
						min(chunkStart + 10, amount)):
				xs.append(x)
				x += self.font_pixels[0] + self.letter_gaps[0]
			x += self.chunkGap
		return xs

	def has_associated_structures(self, line):
		if hasattr(line, 'matchMaps') \
		and [mol for mol in line.matchMaps.keys() if not mol.__destroyed__]:
			return True
		return False

	def hideHeaders(self, headers, pushDown=0, delIndex=None):
		self.top_y += pushDown
		if self.prev_block:
			newLabelWidth = self.prev_block.label_width
		else:
			# assuming parent function passes us a continuous block
			delIndex = self.line_index[headers[0]]
			del self.lines[delIndex:delIndex+len(headers)]
			for line in headers:
				del self.line_index[line]
			for line in self.lines[delIndex:]:
				self.line_index[line] -= len(headers)
			newLabelWidth = self.find_label_width(self.font,
							self.emphasis_font)
		labelChange = newLabelWidth - self.label_width
		self.label_width = newLabelWidth

		for rulerText in self.ruler_texts:
			self.main_scene.move(rulerText, labelChange, pushDown)
		self.bottom_ruler_y += pushDown

		self._moveLines(self.lines[:delIndex], labelChange, 0, pushDown)

		for line in headers:
			labelText = self.label_texts[line]
			del self.label_texts[line]
			self.label_scene.delete(labelText)

			lineItems = self.lineItems[line]
			del self.lineItems[line]
			for item in lineItems:
				if item is not None:
					item.delete()
			del self.itemAuxInfo[line]
		pull = len(headers) * (self.font_pixels[1]
							+ self.letter_gaps[1])
		pushDown -= pull
		self._moveLines(self.lines[delIndex:], labelChange, 0, pushDown)
		self._moveTree(pushDown)

		self.label_width = newLabelWidth
		self.bottom_y += pushDown
		if self.next_block:
			self.next_block.hideHeaders(headers, pushDown=pushDown,
							delIndex=delIndex)

	def highlightName(self, line):
		if self.highlightedName:
			self.label_scene.itemconfigure(self.highlightedName,
				fill=self.normal_label_color)
		self.highlightedName = text = self.label_texts[line]
		self.label_scene.itemconfigure(text, fill='red')
		if self.next_block:
			self.next_block.highlightName(line)

	def _label_font(self, line):
        """TODO
		if self.has_associated_structures(line):
			return self.emphasis_font
        """
		return self.font

	def _largeAlignment(self):
		return len(self.seqs) * len(self.seqs[0]) >= 250000
    """

	def layout_ruler(self, rerule=False):
		if rerule:
			for text in self.ruler_texts:
				self.main_scene.removeItem(text)
		self.ruler_texts = []
		if not self.show_ruler:
			self.bottom_ruler_y = self.top_y
			return
		x = self._left_seqs_edge() + self.font_pixels[0]/2
		y = self.top_y + self.font_pixels[1] + self.letter_gaps[1]

		end = min(self.seq_offset + self.line_width, len(self.alignment.seqs[0]))
		for chunkStart in range(self.seq_offset, end, 10):
			text = self.main_scene.create_text(x, y, anchor='s',
				font=self.font, text="%d" % (chunkStart+1))
			self.ruler_texts.append(text)
			x += self.chunkGap + 10 * (self.font_pixels[0]
							+ self.letter_gaps[0])
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

        from PyQt5.QtWidgets import QGraphicsSimpleTextItem
		text = QGraphicsSimpleTextItem(seq_name(line, self.prefs))
        text.setFont(self._label_font(line))
        text.setBrush(self._brush(label_color, text))
        self.label_scene.addItem(text)
        text.moveBy(x, y)
		self.label_texts[line] = text
        """TODO
		if self.has_associated_structures(line):
			self._colorizeLabel(line)
		bindings = self.label_bindings[line]
		if bindings:
			for eventType, function in bindings.items():
				self.label_scene.tag_bind(text,
							eventType, function)
        """
		colorFunc = self._colorFunc(line)
		lineItems = []
		itemAuxInfo = []
		xs = self._getXs(end - self.seq_offset)
		if self._largeAlignment():
			resStatus = hasattr(line, "matchMaps") and line.matchMaps
		else:
			resStatus = line in self.seqs or adding
		for i in range(end - self.seq_offset):
			item = self.makeItem(line, self.seq_offset + i, xs[i],
				y, half_x, left_rect_off, right_rect_off, colorFunc)
			if resStatus:
				self._assocResBind(item, line, self.seq_offset + i)
			lineItems.append(item)
			itemAuxInfo.append((xs[i], y))

		self.lineItems[line] = lineItems
		self.itemAuxInfo[line] = itemAuxInfo
		if line_index is None:
			self.bottom_y += self.font_pixels[1] + self.letter_gaps[1]

		numberings = [None, None]
		if line.numberingStart != None:
			for numbering in range(2):
				if self.show_numberings[numbering]:
					numberings[numbering] = \
					self._makeNumbering(line, numbering)
		self.numberingTexts[line] = numberings

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
				(y + len(self.lines) - len(self.seqs)) *
				(self.font_pixels[1] + self.letter_gaps[1]))
		x = xFunc(node.xPos, node.xDelta)
		y = yFunc(node.yPos)
		lines = self.treeItems['lines']
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
		self.treeItems['boxes'].append(box)
		if not nodesShown:
			self.label_scene.itemconfigure(box, state='hidden')
		for sn in node.subNodes:
			self._layoutTree(treeInfo, sn, callback, nodesShown, x)
    """

	def _left_seqs_edge(self):
		return self.label_width + self.letter_gaps[0] + self.numbering_widths[0]

    """TODO
	def makeItem(self, line, offset, x, y, half_x,
					left_rect_off, right_rect_off, colorFunc):
		if hasattr(line, 'depictionVal'):
			info = line.depictionVal(offset)
		else:
			info = line[offset]
		if isinstance(info, basestring):
			if len(info) > 1:
				left = x + left_rect_off
				right = x + right_rect_off
				bottom = y
				top = y - self.font_pixels[1]
				return LineItem(info, self.main_scene, left, right, top, bottom,
					colorFunc(line, offset))
			return LineItem('text', self.main_scene, x, y, anchor='s',
				font=self.font, fill=colorFunc(line, offset), text=info)
		if info != None and info > 0.0:
			topRect = y - info * self.font_pixels[1]
			return LineItem('rectangle', self.main_scene, x + left_rect_off,
							topRect, x + right_rect_off, y, width=0,
							outline="", fill=colorFunc(line, offset))
		return None

	def _makeNumbering(self, line, numbering):
		n = self._computeNumbering(line, numbering)
		x, y = self.itemAuxInfo[line][-1]
		if numbering == 0:
			item = self.main_scene.create_text(self._left_seqs_edge(),
				y, anchor='se', font=self.font, text="%d " % n)
		else:
			item = self.main_scene.create_text(x +
				self.base_layout_info()[0], y, anchor='sw',
				font=self.font, text="  %d" % n)
		return item

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

	def _mouseResidue(self, enter, seq=None, index=None):
		if enter:
			self._mouseID = self.main_scene.after(300,
					lambda: self._showResidue(seq, index))
		else:
			if self._mouseID:
				if self._mouseID == "done":
					# only clear the status line if we've
					# put # a residue ID in it previously...
					self.status_func("")
				else:
					self.main_scene.after_cancel(
								self._mouseID)
				self._mouseID = None

	def _moveLines(self, lines, overLabel, overNumber, down):
		over = overLabel + overNumber
		for line in lines:
			self.label_scene.move(self.label_texts[line], 0, down)

			lnum, rnum = self.numberingTexts[line]
			if lnum:
				self.main_scene.move(lnum, overLabel, down)
			if rnum:
				self.main_scene.move(rnum, over, down)
			for item in self.lineItems[line]:
				if item is not None:
					item.move(over, down)
			itemAuxInfo = []
			for oldx, oldy in self.itemAuxInfo[line]:
				itemAuxInfo.append((oldx+over, oldy+down))
			self.itemAuxInfo[line] = itemAuxInfo
			if self.labelRects.has_key(line):
				self.label_scene.move(self.labelRects[line],
								0, down)
	def _moveTree(self, down):
		for itemType, itemList in self.treeItems.items():
			for item in itemList:
				self.label_scene.move(item, 0, down)

	def numBlocks(self):
		if self.next_block:
			return self.next_block.numBlocks() + 1
		return 1

	def pos(self, x, bound=None, y=None):
        """
		"""return 'sequence' position of x"""
        """
		if y is not None and self.next_block \
		and y > self.bottom_y + self.blockGap:
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
			(10 * (self.font_pixels[0] + self.letter_gaps[0])
			+ self.chunkGap))
		chunkX = x - self._left_seqs_edge() - chunk * (
			10 * (self.font_pixels[0] + self.letter_gaps[0])
			+ self.chunkGap)
		chunkOffset = int(chunkX / (self.font_pixels[0]
							+ self.letter_gaps[0]))
		offset = 10 * chunk + min(chunkOffset, 10)
		myLineWidth = min(self.line_width,
					len(self.seqs[0]) - self.seq_offset)
		if offset >= myLineWidth:
			if bound == "left":
				if self.next_block:
					return self.seq_offset + myLineWidth
				return None
			elif bound == "right":
				return self.seq_offset + myLineWidth - 1
			return None
		offset = 10 * chunk + min(chunkOffset, 9)
		rightEdge = self._left_seqs_edge() + \
			chunk * (10 * (self.font_pixels[0]
				+ self.letter_gaps[0]) + self.chunkGap) + \
			(chunkOffset + 1) * (self.font_pixels[0] +
					self.letter_gaps[0])
		if chunkOffset >= 10 or rightEdge - x < self.letter_gaps[0]:
			# in gap
			if bound == "left":
				return self.seq_offset + offset + 1
			elif bound == "right":
				return self.seq_offset + offset
			return None
		# on letter
		return self.seq_offset + offset

	def recolor(self, seq):
		if self.next_block:
			self.next_block.recolor(seq)

		colorFunc = self._colorFunc(seq)

		for i, lineItem in enumerate(self.lineItems[seq]):
			if lineItem is None:
				continue
			lineItem.configure(fill=colorFunc(seq, self.seq_offset + i))
		
	def refresh(self, seq, left, right):
		if self.seq_offset + self.line_width <= right:
			self.next_block.refresh(seq, left, right)
		if left >= self.seq_offset + self.line_width:
			return
		myLeft = max(left - self.seq_offset, 0)
		myRight = min(right - self.seq_offset, self.line_width - 1)

		half_x, left_rect_off, right_rect_off = self.base_layout_info()
		lineItems = self.lineItems[seq]
		itemAuxInfo = self.itemAuxInfo[seq]
		if self._largeAlignment():
			resStatus = hasattr(seq, "matchMaps") and seq.matchMaps
		else:
			resStatus = seq in self.seqs
		colorFunc = self._colorFunc(seq)
		for i in range(myLeft, myRight+1):
			lineItem = lineItems[i]
			if lineItem is not None:
				lineItem.delete()
			x, y = itemAuxInfo[i]
			lineItems[i] = self.makeItem(seq, self.seq_offset + i,
						x, y, half_x, left_rect_off,
						right_rect_off, colorFunc)
			if resStatus:
				self._assocResBind(lineItems[i], seq,
							self.seq_offset + i)
		if self.show_numberings[0] and seq.numberingStart != None \
							and myLeft == 0:
			self.main_scene.delete(self.numberingTexts[seq][0])
			self.numberingTexts[seq][0] = self._makeNumbering(seq,0)
		if self.show_numberings[1] and seq.numberingStart != None \
					and myRight == self.line_width - 1:
			self.main_scene.delete(self.numberingTexts[seq][1])
			self.numberingTexts[seq][1] = self._makeNumbering(seq,1)
		
	def relativeY(self, rawY):
        """
		"""return the y relative to the block the y is in"""
        """
		if rawY < self.top_y:
			if not self.prev_block:
				return 0
			else:
				return self.prev_block.relativeY(rawY)
		if rawY > self.bottom_y + self.blockGap:
			if not self.next_block:
				return self.bottom_y - self.top_y
			else:
				return self.next_block.relativeY(rawY)
		return min(rawY - self.top_y, self.bottom_y - self.top_y)
			
	def realign(self, prevLen):
        """
		"""sequences globally realigned"""
"""

		if shouldWrap(len(self.seqs), self.prefs):
			blockEnd = self.seq_offset + self.line_width
			prev_blockLen = min(prevLen, blockEnd)
			curBlockLen = min(len(self.seqs[0]), blockEnd)
		else:
			blockEnd = len(self.seqs[0])
			self.line_width = blockEnd
			prev_blockLen = prevLen
			curBlockLen = blockEnd

		half_x, left_rect_off, right_rect_off = self.base_layout_info()

		numUnchanged = min(prev_blockLen, curBlockLen) - self.seq_offset
		for line in self.lines:
			lineItems = self.lineItems[line]
			itemAuxInfo = self.itemAuxInfo[line]
			colorFunc = self._colorFunc(line)
			if self._largeAlignment():
				resStatus = hasattr(line, "matchMaps") and line.matchMaps
			else:
				resStatus = line in self.seqs
			for i in range(numUnchanged):
				item = lineItems[i]
				if item is not None:
					item.delete()
				x, y = itemAuxInfo[i]
				lineItems[i] = self.makeItem(line,
					self.seq_offset + i, x, y, half_x,
					left_rect_off, right_rect_off, colorFunc)
				if resStatus:
					self._assocResBind(lineItems[i], line,
							self.seq_offset + i)

		if curBlockLen < prev_blockLen:
			# delete excess items
			self.layout_ruler(rerule=True)
			for line in self.lines:
				lineItems = self.lineItems[line]
				for i in range(curBlockLen, prev_blockLen):
					item = lineItems[i - self.seq_offset]
					if item is None:
						continue
					item.delete()
				start = curBlockLen - self.seq_offset
				end = prev_blockLen - self.seq_offset
				lineItems[start:end] = []
				self.itemAuxInfo[line][start:end] = []
		elif curBlockLen > prev_blockLen:
			# add items
			self.layout_ruler(rerule=True)
			for line in self.lines:
				if self._largeAlignment():
					resStatus = hasattr(line, "matchMaps") and line.matchMaps
				else:
					resStatus = line in self.seqs
				lineItems = self.lineItems[line]
				itemAuxInfo = self.itemAuxInfo[line]
				x, y = itemAuxInfo[0]
				colorFunc = self._colorFunc(line)
				xs = self._getXs(curBlockLen - self.seq_offset)
				for i in range(prev_blockLen, curBlockLen):
					x = xs[i - self.seq_offset]
					lineItems.append(self.makeItem(line, i,
						x, y, half_x, left_rect_off,
						right_rect_off, colorFunc))
					itemAuxInfo.append((x, y))
					if resStatus:
						self._assocResBind(lineItems[-1], line, i)

		if len(self.seqs[0]) <= blockEnd:
			# no further blocks
			if self.next_block:
				self.next_block.destroy()
				self.next_block = None
		else:
			# more blocks
			if self.next_block:
				self.next_block.realign(prevLen)
			else:
				self.next_block = SeqBlock(self.label_scene,
					self.main_scene, self, self.font,
					self.seq_offset + self.line_width,
					self.lines[:0-len(self.seqs)],
					self.seqs, self.line_width,
					self.label_bindings, self.status_func,
					self.show_ruler, self.tree_balloon,
					self.show_numberings, self.prefs)

	def rowIndex(self, y, bound=None):
        """
		"""Given a relative y, return the row index"""
        """
		relRulerBottom = self.bottom_ruler_y - self.top_y
		if y <= relRulerBottom:
			# in header
			if bound == "top":
				return 0
			elif bound == "bottom":
				if self.prev_block:
					return len(self.lines) - 1
				return None
			return None
		row = int((y - relRulerBottom) /
				(self.font_pixels[1] + self.letter_gaps[1]))
		if row >= len(self.lines):
			# off bottom
			if bound == "top":
				if self.next_block:
					return 0
				return None
			elif bound == "bottom":
				return len(self.lines) - 1
			return None
					
		top_y = relRulerBottom + row * (
					self.font_pixels[1] + self.letter_gaps[1])
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
	
	def setLeftNumberingDisplay(self, showNumbering):
		if showNumbering == self.show_numberings[0]:
			return
		self.show_numberings[0] = showNumbering
		numberedLines = [l for l in self.lines if getattr(l,
					'numberingStart', None) != None]
		if showNumbering:
			if not self.prev_block:
				self.numbering_widths[:] = \
					self.findNumberingWidths(self.font)
			delta = self.numbering_widths[0]
			for rulerText in self.ruler_texts:
				self.main_scene.move(rulerText, delta, 0)
			for line in numberedLines:
				self.numberingTexts[line][0] = \
						self._makeNumbering(line, 0)
			self._moveLines(self.lines, 0, delta, 0)
		else:
			delta = 0 - self.numbering_widths[0]
			for rulerText in self.ruler_texts:
				self.main_scene.move(rulerText, delta, 0)
			for texts in self.numberingTexts.values():
				if not texts[0]:
					continue
				self.main_scene.delete(texts[0])
				texts[0] = None
			self._moveLines(self.lines, 0, delta, 0)
			if not self.next_block:
				self.numbering_widths[0] = 0
		if self.next_block:
			self.next_block.setLeftNumberingDisplay(showNumbering)

	def setRightNumberingDisplay(self, showNumbering):
		if showNumbering == self.show_numberings[1]:
			return
		self.show_numberings[1] = showNumbering
		if showNumbering:
			if not self.prev_block:
				self.numbering_widths[:] = \
					self.findNumberingWidths(self.font)
			numberedLines = [l for l in self.lines if getattr(l,
					'numberingStart', None) != None]
			for line in numberedLines:
				self.numberingTexts[line][1] = \
						self._makeNumbering(line, 1)
		else:
			for texts in self.numberingTexts.values():
				if not texts[1]:
					continue
				self.main_scene.delete(texts[1])
				texts[1] = None
			if not self.next_block:
				self.numbering_widths[1] = 0
		if self.next_block:
			self.next_block.setRightNumberingDisplay(showNumbering)

	def setRulerDisplay(self, show_ruler, pushDown=0):
		if show_ruler == self.show_ruler:
			return
		self.show_ruler = show_ruler
		self.top_y += pushDown
		self.bottom_y += pushDown
		pull = self.font_pixels[1] + self.letter_gaps[1]
		if show_ruler:
			pushDown += pull
			self.layout_ruler()
		else:
			for text in self.ruler_texts:
				self.main_scene.delete(text)
			pushDown -= pull
			self.bottom_ruler_y = self.top_y
			self.bottom_y -= pull
		self._moveLines(self.lines, 0, 0, pushDown)
		self._moveTree(pushDown)
		if self.next_block:
			self.next_block.setRulerDisplay(show_ruler,
							pushDown=pushDown)

	def _showResidue(self, aseq, index):
		self._mouseID = "done"
		ungapped = aseq.gapped2ungapped(index)
		if ungapped is None:
			statusText = "gap"
		elif hasattr(aseq, 'matchMaps'):
			residues = []
			for matchMap in aseq.matchMaps.values():
				try:
					residues.append(matchMap[ungapped])
				except KeyError:
					continue
			if residues:
				statusText = ", ".join([chimeraLabel(r,
					modelName=True, style="simple") for r in residues])
				if len(statusText) > 100:
					statusText = ", ".join([chimeraLabel(r,
						modelName=False, style="simple") for r in residues])
				if len(statusText) > 100:
					statusText = ", ".join([chimeraLabel(r,
						modelName=False, style="osl") for r in residues])
			else:
				statusText = "no corresponding structure residue"
		else:
			if aseq.numberingStart is None:
				off = 1
			else:
				off = aseq.numberingStart
			statusText = "sequence position %d" % (ungapped+off)

		self.status_func(statusText, blankAfter=0)

	def showHeaders(self, headers, pushDown=0):
		self.top_y += pushDown
		if self.prev_block:
			newLabelWidth = self.prev_block.label_width
			insertIndex = len(self.lines) - len(self.seqs) - len(
								headers)
		else:
			insertIndex = len(self.lines) - len(self.seqs)
			self.lines[insertIndex:insertIndex] = headers
			for seq in self.seqs:
				self.line_index[seq] += len(headers)
			for i in range(len(headers)):
				self.line_index[headers[i]] = insertIndex + i
			newLabelWidth = self.find_label_width(self.font,
							self.emphasis_font)
		labelChange = newLabelWidth - self.label_width
		self.label_width = newLabelWidth

		for rulerText in self.ruler_texts:
			self.main_scene.move(rulerText, labelChange, pushDown)
		self.bottom_ruler_y += pushDown

		self._moveLines(self.lines[:insertIndex], labelChange, 0,
								pushDown)

		for i in range(len(headers)):
			self._layout_line(headers[i], self.header_label_color,
						line_index=insertIndex+i)
		push = len(headers) * (self.font_pixels[1]
							+ self.letter_gaps[1])
		pushDown += push
		self._moveLines(self.lines[insertIndex+len(headers):],
						labelChange, 0, pushDown)
		self._moveTree(pushDown)
		self.label_width = newLabelWidth
		self.bottom_y += pushDown
		if self.next_block:
			self.next_block.showHeaders(headers, pushDown=pushDown)

	def showNodes(self, show):
		if show:
			state = 'normal'
		else:
			state = 'hidden'
		for box in self.treeItems['boxes']:
			self.label_scene.itemconfigure(box, state=state)
		if self.next_block:
			self.next_block.showNodes(show)

	def showTree(self, treeInfo, callback, nodesShown, active=None):
		for box in self.treeItems['boxes']:
			self.tree_balloon.tagunbind(self.label_scene, box)
		for treeItems in self.treeItems.values():
			while treeItems:
				self.label_scene.delete(treeItems.pop())
		if self.next_block:
			self.next_block.showTree(treeInfo, callback, nodesShown)
		if not treeInfo:
			return
		self.treeNodeMap = {'active': active}
		self._layoutTree(treeInfo, treeInfo['tree'], callback,
								nodesShown)

	def updateNumberings(self):
		numberedLines = [l for l in self.lines if getattr(l,
					'numberingStart', None) != None]
		for line in numberedLines:
			for i in range(2):
				nt = self.numberingTexts[line][i]
				if not nt:
					continue
				self.main_scene.delete(nt)
				self.numberingTexts[line][i] = \
						self._makeNumbering(line, i)
		if self.next_block:
			self.next_block.updateNumberings()

def shouldWrap(numSeqs, prefs):
	if numSeqs == 1:
		prefix = SINGLE_PREFIX
	else:
		prefix = ""
	if prefs[prefix + WRAP_IF]:
		if numSeqs <= prefs[prefix + WRAP_THRESHOLD]:
			return 1
		else:
			return 0
	elif prefs[prefix + WRAP]:
		return 1
	return 0
"""

def seq_name(seq, prefs):
    """TODO
	return ellipsis_name(seq.name, prefs[SEQ_NAME_ELLIPSIS])
    """
	return ellipsis_name(seq.name, 30)

def ellipsis_name(name, ellipsis_threshold):
	if len(name) > ellipsis_threshold:
		half = int(ellipsis_threshold/2)
		return name[0:half-1] + "..." + name[len(name)-half:]
	return name
