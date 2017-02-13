# vim: set expandtab ts=4 sw=4:

"""TODO
from chimera.baseDialog import ModelessDialog
from chimera import replyobj
from CGLtk.color import rgba2tk
from chimera import colorTable
import Pmw
import Tkinter
import os
import chimera
from chimera import selectionOperation, MOTION_STOP
from chimera.selection import ItemizedSelection, currentResidues
import operator
from MAViewer import DEL_ASSOC, MOD_ASSOC, PRE_DEL_SEQS, SEQ_RENAMED
from prefs import SHOW_SEL, SCF_COLOR_STRUCTURES, NEW_REGION_BORDER, \
    NEW_REGION_INTERIOR, SEL_REGION_BORDER, SEL_REGION_INTERIOR
from prefs import REGION_NAME_ELLIPSIS, REGION_BALLOON_ON
from prefs import RB_LAST_USE
"""

class Region:
    def __init__(self, region_browser, name=None, init_blocks=[], shown=True,
            border_rgba=None, interior_rgba=None, name_prefix="",
            cover_gaps=False, source=None):
        self._name = name
        self.name_prefix = name_prefix
        self.region_browser = region_browser
        self.seq_canvas = self.region_browser.seq_canvas
        self.scene = self.seq_canvas.main_scene
        self._border_rgba = border_rgba
        self._interior_rgba = interior_rgba
        self.cover_gaps = cover_gaps
        self.source = source
        self.highlighted = False

        self._items = []
        self.blocks = []
        self._shown = bool(shown)
        self._active = False
        self.sequence = self.associated_with = None
        self.add_blocks(init_blocks, make_cb=False)

    def set_active(self, val):
        if val == self.highlighted:
            return
        self.region_browser._toggle_active(self)

    def get_active(self):
        return self.highlighted

    active = property(get_active, set_active)

    def add_block(self, block):
        self.add_blocks([block])

    def add_blocks(self, block_list, make_cb=True):
        self.blocks.extend(block_list)
        if make_cb:
            self.region_browser._region_size_changed_cb(self)
        if not self.shown:
            return
        kw = self._rect_kw()
        for block in block_list:
            for x1, y1, x2, y2 in self.seq_canvas.bbox_list(cover_gaps=self.cover_gaps, *block):
                """
                if self._disp_border_rgba():
                    # for some reason, tk canvas
                    # rectangle borders are inside the
                    # upper left edges and outside the
                    # lower right
                    x1 -= 1
                    y1 -= 1
                """
                print("Added rect at", x1, y2, x2-x1, y2-1, "with kw", kw)
                #self._items.append(self.scene.addRect(x1, y2, x2-x1, y2-y1, **kw))
                self._items.append(self.scene.addRect(x1, y2, x2-x1, y2-y1))
                if self.name:
                    self._items[-1].setToolTip(str(self))
                if len(self._items) > 1:
                    self._items[-1].setZValue(self._items[-2].zValue()-0.001)
                    continue

                regions = self.region_browser.regions
                above = below = None
                if self in regions:
                    index = regions.index(self)
                    for r in regions[index+1:]:
                        if r._items:
                            above = r
                            break
                    else:
                        for i in range(index-1, -1, -1):
                            r = regions[i]
                            if r._items:
                                below = r
                                break
                else:
                    for r in regions:
                        if r._items:
                            above = r
                            break
                if above:
                    self._items[-1].setZValue(above._items[0].zValue()+1)
                elif below:
                    self._items[-1].setZValue(below._items[0].zValue()-1)
                else:
                    self._items[-1].setZValue(-100000000)

    def _addLines(self, lines):
        pass

    def get_border_rgba(self):
        return self._border_rgba

    def set_border_rgba(self, rgba):
        if self.border_rgba == rgba:
            return
        self._border_rgba = rgba
        # kind of complicated due to highlighting; just redraw
        self.redraw()

    border_rgba = property(get_border_rgba, set_border_rgba)

    def clear(self, make_cb=True):
        if self.blocks:
            self.blocks = []
            for item in self._items:
                self.scene.removeItem(item)
            self._items = []
            if make_cb:
                self.region_browser._region_size_changed_cb(self)

    def contains(self, x, y):
        from PyQt5.QtCore import QPointF
        for item in self.scene.items(QPointF(x, y)):
            if item in self._items:
                return True
        return False

    def dehighlight(self):
        if not self.highlighted:
            return
        self.highlighted = False
        self.redraw()
        """TODO
        rb = self.region_browser
        rb.update_table_cell(self, rb.active_column, contents=False)
        """

    def _del_lines(self, lines):
        blocks = self.blocks
        self.blocks = []
        for block in blocks:
            line1, line2, pos1, pos2 = block
            li1 = self.seq_canvas.lead_block.line_index[line1]
            li2 = self.seq_canvas.lead_block.line_index[line2]
            for nli1 in range(li1, li2+1):
                new_line1 = self.seq_canvas.lead_block.lines[nli1]
                if new_line1 not in lines:
                    break
            else:
                continue
            for nli2 in range(li2, nli1-1, -1):
                new_line2 = self.seq_canvas.lead_block.lines[nli2]
                if new_line2 not in lines:
                    break
            self.blocks.append((new_line1, new_line2, pos1, pos2))

    def destroy(self, rebuild_table=True):
        for item in self._items:
            self.scene.removeItem(item)
        self.blocks = []
        self.sequence = self.associated_with = None
        self.region_browser._region_destroyed_cb(self, rebuild_table=rebuild_table)

    def _disp_border_rgba(self):
        # account for highlighting
        if not self.border_rgba and self.highlighted:
            return (0.0, 0.0, 0.0, 1.0)
        return self.border_rgba

    def highlight(self):
        if self.highlighted:
            return
        self.highlighted = True
        self.redraw()
        """TODO
        rb = self.region_browser
        rb.update_table_cell(self, rb.active_column, contents=True)
        """

    def get_interior_rgba(self):
        return self._interior_rgba

    def set_interior_rgba(self, rgba):
        if rgba == self.interior_rgba:
            return
        self._interior_rgba = rgba
        if not self._items:
            return
        brush = self._items[0].brush()
        from PyQt5.QtCore import Qt
        if rgba:
            from PyQt5.QtGui import QColor
            brush.setColor(QColor(*rgba))
            brush.setStyle(Qt.SolidPattern)
        else:
            brush.setStyle(Qt.NoBrush)
        for item in self._items:
            item.setBrush(brush)

    interior_rgba = property(get_interior_rgba, set_interior_rgba)

    def lower_below(self, other_region):
        if not self._items or not other_region._items:
            return
        self._items[0].setZValue(other_region.zValue()-1)
        for i in range(1, len(self._items)):
            self._items[i].setZValue(self._items[i-1].zValue()-0.001)

    def set_name(self, val):
        if val == self._name:
            return
        if val == None:
            for item in self._items:
                item.setToolTip("")
        self._name = val
        if val:
            for item in self._items:
                item.setToolTip(str(self))

    def get_name(self):
        return self._name

    name = property(get_name, set_name)

    def raise_above(self, other_region):
        if not self._items or not other_region._items:
            return
        self._items[0].setZValue(other_region._items[0].zValue()+1)
        for i in range(1, len(self._items)):
            self._items[i].setZValue(self._items[i-1].zValue()-0.001)

    def _rect_kw(self):
        kw = {}
        from PyQt5.QtGui import QBrush, QPen, QColor
        kw['pen'] = pen = QPen()
        kw['brush'] = brush = QBrush()
        from PyQt5.QtCore import Qt
        if self.interior_rgba is not None:
            brush.setColor(QColor(*self._interior_rgba))
            brush.setStyle(Qt.SolidPattern)
        else:
            brush.setStyle(Qt.NoBrush)
        if self._disp_border_rgba():
            pen.setColor(QColor(*self._disp_border_rgba()))
            if self.highlighted:
                pen.setStyle(Qt.DashLine)
            else:
                pen.setStyle(Qt.SolidLine)
        else:
            pen.setStyle(Qt.NoPen)
        return kw

    def redraw(self):
        for item in self._items:
            self.scene.removeItem(item)
        self._items = []
        blocks = self.blocks
        self.blocks = []
        self.add_blocks(blocks, make_cb=False)

    def remove_last_block(self, destroy_if_empty=False, make_cb=True):
        if self.blocks:
            prev_bboxes = self.seq_canvas.bbox_list(cover_gaps=self.cover_gaps, *self.blocks[-1])
            self.blocks = self.blocks[:-1]
            for item in self._items[0-len(prev_bboxes):]:
                self.scene.removeItem(item)
            self._items = self._items[:0-len(prev_bboxes)]
        if destroy_if_empty and not self.blocks:
            self.destroy()
        elif make_cb:
            self.region_browser._region_size_changed_cb(self)

    def get_rmsd(self):
        num_d = sum_d2 = 0
        from chimerax.core.geometry import distance_squared
        for block in self.blocks:
            line1, line2, pos1, pos2 = block
            all_seqs = self.seq_canvas.alignment.seqs
            try:
                si1 = all_seqs.index(line1)
            except ValueError:
                si1 = 0
            try:
                si2 = all_seqs.index(line2)
            except ValueError:
                continue
            match_info = []
            seqs = all_seqs[si1:si2+1]
            for seq in seqs:
                for match_map in seq.match_maps.values():
                    match_info.append((seq, match_map))
            if len(match_info) < 2:
                continue
            for pos in range(pos1, pos2+1):
                atoms = []
                for seq, match_map in match_info:
                    ungapped = seq.gapped_to_ungapped(pos)
                    if ungapped == None or ungapped not in match_map:
                        continue
                    res = match_map[ungapped]
                    atom = res.principal_atom
                    if atom:
                        atoms.append(atom)
                for i, a1 in enumerate(atoms):
                    for a2 in atoms[i+1:]:
                        num_d += 1
                        sum_d2 += distance_squared(a1.scene_coord, a2.scene_coord)
        if num_d:
            from math import sqrt
            return sqrt(sum_d2/num_d)
        return None
    
    rmsd = property(get_rmsd)

    def set_cover_gaps(self, cover):
        if cover != self.cover_gaps:
            self.cover_gaps = cover
            self.redraw()

    def set_shown(self, val):
        if bool(val) == self._shown:
            return
        self._shown = bool(val)
        self.redraw()
        rb = self.region_browser
        rb.update_table_cell(self, rb.shown_column, contents=self.shown)

    def get_shown(self):
        return self._shown

    shown = property(get_shown, set_shown)

    def __str__(self):
        if self.name:
            return self.name_prefix + self.name
        if not self.blocks:
            return self.name_prefix + "<empty>"
        line1, line2, pos1, pos2 = self.blocks[0]
        if line1 != line2:
            base = "%s..%s " % (line1.name, line2.name)
        else:
            base = line1.name + " "
        if pos1 != pos2:
            base += "[%d-%d]" % (pos1+1, pos2+1)
        else:
            base += "[%d]" % (pos1+1)

        if len(self.blocks) > 1:
            base += " + %d other block" % (len(self.blocks) - 1)
        if len(self.blocks) > 2:
            base += "s"
        return self.name_prefix + base

    def update_last_block(self, block):
        self.remove_last_block(make_cb=False)
        self.add_blocks([block])

class RegionBrowser:

    """
    PRED_HELICES_REG_NAME = "predicted helices"
    PRED_STRANDS_REG_NAME = "predicted strands"
    PRED_SS_REG_NAMES = [PRED_HELICES_REG_NAME, PRED_STRANDS_REG_NAME]
    ACTUAL_HELICES_REG_NAME = "structure helices"
    ACTUAL_STRANDS_REG_NAME = "structure strands"
    ACTUAL_SS_REG_NAMES = [ACTUAL_HELICES_REG_NAME, ACTUAL_STRANDS_REG_NAME]
    SS_REG_NAMES = PRED_SS_REG_NAMES + ACTUAL_SS_REG_NAMES
    """

    def __init__(self, tool_window, seq_canvas):
        self.tool_window = tool_window
        self.seq_canvas = seq_canvas
        seq_canvas.main_scene.mousePressEvent = self._mouse_down_cb
        seq_canvas.main_scene.mouseReleaseEvent = self._mouse_up_cb
        seq_canvas.main_scene.mouseDoubleClickEvent = \
            lambda e, s=self: s._mouse_up_cb(e, double=True)
        seq_canvas.main_scene.mouseMoveEvent = self._mouse_drag_cb
        self._drag_lines = []
        self._drag_region = None
        self._prev_drag = None
        self._bboxes = []
        self._after_id = None
        self._highlighted_region = None
        self.regions = []
        self.associated_regions = {}
        self.rename_dialogs = {}
        self._scf_dialog = None
        self._mod_assoc_handler_id = self._motion_stop_handler_id = None
        self.sequence_regions = { None: set() }
        self._cur_region = None
        """
        ModelessDialog.__init__(self)
        """

    def clearRegions(self, doSingleSeqRegions=True):
        if doSingleSeqRegions:
            for region in self.regions[:]:
                region.destroy()
            self.associated_regions.clear()
            self.sequence_regions = { None: set() }
        else:
            singleSeqRegions = set()
            for seq, regions in self.sequence_regions.items():
                if seq is not None:
                    singleSeqRegions.update(regions)
            for region in self.regions[:]:
                if region not in singleSeqRegions:
                    region.destroy()

    def copyRegion(self, region, name=None, **kw):
        if not region:
            self.seq_canvas.mav.status("No active region",
                                color="red")
            return
        if not isinstance(region, Region):
            for r in region:
                self.copyRegion(r)
            return
        if name is None:
            initialName = "Copy of " + unicode(region)
        else:
            initialName = name
        seq = region.sequence
        copy = self.new_region(name=initialName,
                blocks=region.blocks, fill=region.interior_rgba,
                outline=region.border_rgba, sequence=seq,
                cover_gaps=region.cover_gaps, **kw)
        self.regionListing.select([copy])
        if name is None:
            self.renameRegion(copy)
        return copy

    def cur_region(self):
        return self._cur_region

    def deleteRegion(self, region, rebuild_table=True):
        if not region:
            self.seq_canvas.mav.status("No active region",
                                color="red")
            return
        if not isinstance(region, Region):
            for r in region:
                self.deleteRegion(r, rebuild_table=(r == region[-1]))
            return
        if region == self.get_region("ChimeraX selection"):
            self.seq_canvas.mav.status(
                "Cannot delete ChimeraX selection region",
                color="red")
        else:
            assoc = region.associated_with
            if assoc:
                regions = self.associated_regions[assoc]
                regions.remove(region)
                if not regions:
                    del self.associated_regions[assoc]
            seq = region.sequence
            regions = self.sequence_regions[seq]
            regions.remove(region)
            region.destroy(rebuild_table=rebuild_table)
            if seq and not regions:
                del self.sequence_regions[seq]
                self.seqRegionMenu.setitems(self._regMenuOrder())
                if rebuild_table:
                    self.seqRegionMenu.invoke(0)

    def destroy(self):
        self.regionListing.destroy()
        self.seq_canvas.mav.triggers.deleteHandler(DEL_ASSOC,
                            self._delAssocHandlerID)
        if self._mod_assoc_handler_id:
            self.seq_canvas.mav.triggers.deleteHandler(MOD_ASSOC,
                            self._mod_assoc_handler_id)
        if self._motion_stop_handler_id:
            chimera.triggers.deleteHandler(MOTION_STOP,
                            self._motion_stop_handler_id)
        self.seq_canvas.mav.triggers.deleteHandler(PRE_DEL_SEQS,
            self._pre_del_seqs_handler_id)
        self.seq_canvas.mav.triggers.deleteHandler(SEQ_RENAMED,
                            self._seqRenamedHandlerID)
        if hasattr(self,'_selChangeHandler') and self._selChangeHandler:
            chimera.triggers.deleteHandler("selection changed",
                            self._selChangeHandler)
        if self._scf_dialog:
            self._scf_dialog.destroy()
            self._scf_dialog = None
        for rd in self.rename_dialogs.values():
            rd.destroy()
        self.rename_dialogs.clear()
        ModelessDialog.destroy(self)

    def fillInUI(self, parent):
        self.Close()
        row = 0
        browseFrame = Tkinter.Frame(parent)
        browseFrame.grid(row=row, column=0, columnspan=3, sticky='nsew')
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        self.seqRegionMenu = Pmw.OptionMenu(browseFrame, items=["entire alignment"],
            initialitem=0, labelpos="w", label_text="Regions applicable to:",
            command=self._rebuildListing)
        from CGLtk.Table import SortableTable
        self.regionListing = SortableTable(browseFrame, allowUserSorting=False)
        prefs = self.seq_canvas.mav.prefs
        last = prefs[RB_LAST_USE]
        from time import time
        now = prefs[RB_LAST_USE] = time()
        if last is None or now - last > 777700: # about 3 months
            aTitle, sTitle = "Active", "Shown"
        else:
            aTitle, sTitle = "A", "S"
        self.active_column = self.regionListing.addColumn(aTitle,
                                "active", format=bool)
        self.shown_column = self.regionListing.addColumn(sTitle,
                                "shown", format=bool)
        self.regionListing.addColumn(u"\N{BLACK MEDIUM SQUARE}", "interior_rgba",
            format=(True, False), color="purple")
        self.regionListing.addColumn(u"\N{BALLOT BOX}", "border_rgba",
            format=(True, False), color="forest green")
        self.regionListing.addColumn("Name",
                lambda r, prefs=prefs: region_name(r, prefs), anchor='w')
        self.rmsdCol = self.regionListing.addColumn("RMSD", "rmsd", format="%.3f")
        def blocks2val(region, index, func):
            blocks = region.blocks
            if not blocks:
                return None
            return func([b[index] for b in blocks])+1
        self.startCol = self.regionListing.addColumn("Start",
                lambda r: blocks2val(r, 2, min), format="%d", display=False)
        self.endCol = self.regionListing.addColumn("End",
                lambda r: blocks2val(r, 3, max), format="%d", display=False)
        self.regionListing.setData(self.regions)
        self.regionListing.launch(browseCmd=self._listingCB)
        self.regionButtons = {}
        for i, butName in enumerate(('raise', 'lower', 'copy',
                            'rename', 'delete', 'info')):
            self.regionButtons[butName] = Tkinter.Button(
                browseFrame, text=butName.capitalize(),
                command=lambda cmd=getattr(self,
                butName+"Region"): cmd(self.selected()), pady=0)
            self.regionButtons[butName].grid(row=2, column=i)
            browseFrame.columnconfigure(i, weight=1)
        self.seqRegionMenu.grid(row=0, column=0, columnspan=i+1)
        self.regionListing.grid(row=1, column=0, columnspan=i+1, sticky='nsew')
        browseFrame.rowconfigure(1, weight=1)
        row += 1

        self.sourceArea = Tkinter.LabelFrame(parent, text="Data source")
        subtitle = Tkinter.Label(self.sourceArea, text="Show these kinds"
            " of regions...")
        from CGLtk.Font import shrinkFont
        shrinkFont(subtitle)
        subtitle.grid(row=0, column=0)
        self.sourceButtonArea = Tkinter.Frame(self.sourceArea)
        self.sourceButtonArea.grid(row=1, column=0)
        biVar = Tkinter.IntVar(parent)
        biVar.set(True)
        biBut = Tkinter.Checkbutton(self.sourceButtonArea, text="built-in",
            variable=biVar, command=self._rebuildListing)
        self.sourceControls = { None: (biVar, biBut) }
        self.sourceArea.grid(row=row, column=0, columnspan=3)
        self.sourceArea.grid_remove()

        row += 1

        afterTableRow = row
        Tkinter.Label(parent, text="Choosing in table:"
            ).grid(row=row, column=0, rowspan=3, sticky='e')
        self.activateVar = Tkinter.IntVar(parent)
        self.activateVar.set(True)
        Tkinter.Checkbutton(parent, variable=self.activateVar,
            text="activates").grid(row=row, column=1, sticky='w')
        row += 1
        self.showVar = Tkinter.IntVar(parent)
        self.showVar.set(True)
        Tkinter.Checkbutton(parent, variable=self.showVar,
            text="shows").grid(row=row, column=1, sticky='w')
        row += 1
        self.hideVar = Tkinter.IntVar(parent)
        self.hideVar.set(True)
        Tkinter.Checkbutton(parent, variable=self.hideVar,
            text="hides others except missing structure, ChimeraX selection"
            ).grid(row=row, column=1, columnspan=3, sticky='w')
        from chimera.tkoptions import BooleanOption
        class CoverGapsOption(BooleanOption):
            attribute = "cover_gaps"
        f = Tkinter.Frame(parent)
        f.grid(row=afterTableRow, column=2, rowspan=2)
        self.cover_gapsOption = CoverGapsOption(f, 0,
            "Include gaps", None, self._cover_gapsCB)

        self._sel_change_from_self = False
        self._firstChimeraShow = True
        if self.seq_canvas.mav.prefs[SHOW_SEL]:
            self._showSelCB()

        cb = lambda e, s=self: s.deleteRegion(s.selected())
        parent.winfo_toplevel().bind('<Delete>', cb)
        parent.winfo_toplevel().bind('<BackSpace>', cb)

        self._delAssocHandlerID = self.seq_canvas.mav.triggers.addHandler(
                    DEL_ASSOC, self._delAssocCB, None)

    def get_region(self, name, sequence=False, **kw):
        try:
            create = kw['create']
            del kw['create']
        except KeyError:
            create = False
        if sequence is False:
            regions = self.regions
        else:
            regions = self.sequence_regions.get(sequence, [])
        for region in regions:
            if region.name == name:
                return region
        if not create:
            return None
        if sequence is not False:
            kw['sequence'] = sequence
        if 'cover_gaps' not in kw:
            kw['cover_gaps'] = False
        return self.new_region(name=name, **kw)

    def highlight(self, region, select_on_structures=True):
        if self._highlighted_region:
            self._highlighted_region.dehighlight()
        if region:
            region.highlight()
            if select_on_structures and region != self.get_region("ChimeraX selection"):
                self._select_on_structures(region)
        self._highlighted_region = region

    def infoRegion(self, region):
        if not region:
            self.seq_canvas.mav.status("No active region",
                                color="red")
            return
        if not isinstance(region, Region):
            for r in region:
                self.infoRegion(r)
            return

        # gapped
        info = unicode(region) + \
            " region covers positions:\n"
        info += "\talignment numbering: " + ", ".join(
            ["%d-%d" % (b[-2]+1, b[-1]+1) for b in region.blocks]) + "\n"

        # ungapped
        seqs = self.seq_canvas.seqs
        structInfo = {}
        for i, seq in enumerate(seqs):
            blocks = []
            for line1, line2, pos1, pos2 in region.blocks:
                try:
                    index1 = seqs.index(line1)
                except ValueError:
                    index1 = -1
                try:
                    index2 = seqs.index(line2)
                except ValueError:
                    index2 = -1
                if index1 <= i <= index2:
                    for p1 in range(pos1, pos2+1):
                        if not seq.isGap(p1):
                            break
                    else:
                        continue
                    for p2 in range(pos2, pos1-1, -1):
                        if not seq.isGap(p2):
                            break
                    else:
                        continue
                    u1, u2 = [seq.gapped2ungapped(p) for p in (p1, p2)]
                    blocks.append((u1, u2))
                    for mol, matches in seq.match_maps.items():
                        for r1 in range(u1, u2+1):
                            try:
                                m1 = matches[r1]
                            except KeyError:
                                continue
                            if m1:
                                break
                        else:
                            continue
                        for r2 in range(u2, u1-1, -1):
                            try:
                                m2 = matches[r2]
                            except KeyError:
                                continue
                            if m2:
                                break
                        else:
                            continue
                        residues = (m1, m2)
                        if mol in structInfo:
                            structInfo[mol][-1].append(residues)
                        else:
                            structInfo[mol] = (i, [residues])
            if blocks:
                if seq.numberingStart is None:
                    off = 1
                else:
                    off = seq.numberingStart
                info += "\t" + seq.name + ": " + ", ".join(
                    ["%d-%d" % (p1+off, p2+off) for p1, p2 in blocks]) + "\n"

        # associated structures
        if structInfo:
            sortableRanges = structInfo.values()
            sortableRanges.sort()
            info += unicode(region) + " region's associated structures:\n"
            for i, resRanges in sortableRanges:
                info += "\t%s: " % resRanges[0][0].molecule + ", ".join(
                    [u"%s \N{LEFT RIGHT ARROW} %s" % (r1, r2)
                    for r1, r2 in resRanges]) + "\n"
            info += "\n"
        else:
            info += unicode(region) + " region has no associated structures\n\n"
        replyobj.info(info)
        self.seq_canvas.mav.status("Region info reported in reply log")
        from chimera import dialogs
        dialogs.display("reply")

    def lastDrag(self):
        return self._prev_drag

    def loadScfCB(self, okayed, dialog):
        if not okayed:
            return
        self.seq_canvas.mav.prefs[SCF_COLOR_STRUCTURES] = \
                        dialog.colorStructureVar.get()

        for path in dialog.getPaths():
            self.loadScfFile(path,
                self.seq_canvas.mav.prefs[SCF_COLOR_STRUCTURES])
        
    def loadScfFile(self, path, colorStructures=True):
        if path is None:
            if not self._scf_dialog:
                self._scf_dialog = ScfDialog(
                        self.seq_canvas.mav.prefs[
                        SCF_COLOR_STRUCTURES],
                        command=self.loadScfCB)
            self._scf_dialog.enter()
            return

        seqs = self.seq_canvas.seqs
        from OpenSave import osOpen
        scfFile = osOpen(path)
        lineNum = 0
        regionInfo = {}
        for line in scfFile.readlines():
            lineNum += 1
            line.strip()
            if not line or line[0] == '#' \
            or line.startswith('//'):
                continue
            for commentIntro in ['//', '#']:
                commentPos = line.find(commentIntro)
                if commentPos >= 0:
                    break
            if commentPos >= 0:
                comment = line[commentPos
                    + len(commentIntro):].strip()
                line = line[:commentPos].strip()
            else:
                comment = None

            try:
                pos1, pos2, seq1, seq2, r, g, b = map(
                        int, line.split())
                if seq1 == -1:
                    # internal to jevtrace/webmol
                    continue
                if seq1 == 0:
                    seq2 = -1
                else:
                    seq1 -= 1
                    seq2 -= 1
            except:
                try:
                    pos, seq, r, g, b = map(int,
                            line.split())
                    pos1 = pos2 = pos
                except:
                    replyobj.error("Bad format for line %d of %s [not 5 or 7 integers]\n" % (lineNum, path))
                    scfFile.close()
                    return
                if seq == 0:
                    seq1 = 0
                    seq2 = -1
                else:
                    seq1 = seq2 = seq - 1
            key = ((r, g, b), comment)
            if key in regionInfo:
                regionInfo[key].append((seqs[seq1],
                    seqs[seq2], pos1, pos2))
            else:
                regionInfo[key] = [(seqs[seq1],
                    seqs[seq2], pos1, pos2)]
        scfFile.close()

        if not regionInfo:
            replyobj.error("No annotations found in %s\n"
                % path)
            return
        for rgbComment, blocks in regionInfo.items():
            rgb, comment = rgbComment
            rgb = map(lambda v: v/255.0, rgb)
            region = self.new_region(name_prefix="Seqsel: ",
                blocks=blocks, name=comment, fill=rgb,
                cover_gaps=True)
            if not colorStructures:
                continue
            c = chimera.MaterialColor(*rgb)
            for res in self.region_residues(region):
                res.ribbonColor = c
                for a in res.atoms:
                    a.color = c
        self.seq_canvas.mav.status("%d scf regions created"
                        % len(regionInfo))
        
    def lowerRegion(self, region, rebuild_table=True):
        if not region:
            self.seq_canvas.mav.status("No active region",
                                color="red")
            return
        if not isinstance(region, Region):
            for r in region:
                self.lowerRegion(r)
            return
        index = self.regions.index(region)
        if index == len(self.regions) - 1:
            return
        self.regions.remove(region)
        self.regions.append(region)
        for higherRegion in self.regions[-2::-1]:
            if higherRegion.blocks and higherRegion.shown:
                region.lower_below(higherRegion)
                break
        if rebuild_table:
            self._rebuildListing()

    def map(self, *args):
        refreshRMSD = lambda *args: self.regionListing.refresh()
        self._mod_assoc_handler_id = self.seq_canvas.mav.triggers.addHandler(
            MOD_ASSOC, refreshRMSD, None)
        self._motion_stop_handler_id = chimera.triggers.addHandler(
            MOTION_STOP, refreshRMSD, None)
        refreshRMSD()

    def moveRegions(self, offset, startOffset=0, exceptBottom=None):
        if not offset:
            return
        for region in self.regions:
            blocks = region.blocks[:]
            region.clear(make_cb=False)
            newBlocks = []
            for l1, l2, p1, p2 in blocks:
                if l2 != exceptBottom:
                    if p1 >= startOffset:
                        p1 += offset
                    if p2 >= startOffset:
                        p2 += offset
                newBlocks.append([l1, l2, p1, p2])
            region.add_blocks(newBlocks, make_cb=False)

    def new_region(self, name=None, blocks=[], fill=None, outline=None,
            name_prefix="", select=False, assoc_with=None, shown=True,
            cover_gaps=True, after="ChimeraX selection", rebuild_table=True,
            session_restore=False, sequence=None, source=None):
        if not name and not name_prefix:
            # possibly first user-dragged region
            for reg in self.regions:
                if not reg.name and not reg.name_prefix:
                    # another user region
                    break
            else:
                self.seq_canvas.mav.status("Use delete/backspace key to remove regions")
        interior = self._get_rgba(fill)
        border = self._get_rgba(outline)
        region = Region(self, init_blocks=blocks, name=name, name_prefix=name_prefix, shown=shown,
                border_rgba=border, interior_rgba=interior, cover_gaps=cover_gaps, source=source)
        if isinstance(after, Region):
            insert_index = self.regions.index(after) + 1
        elif isinstance(after, str):
            try:
                insert_index = [r.name for r in self.regions].index(after) + 1
            except ValueError:
                insert_index = 0
        else:
            insert_index = 0
        self.regions.insert(insert_index, region)
        if sequence not in self.sequence_regions:
            self.sequence_regions[sequence] = set([region])
            """TODO
            index = self.seqRegionMenu.index(Pmw.SELECT)
            self.seqRegionMenu.setitems(self._regMenuOrder(), index=index)
            """
        else:
            self.sequence_regions[sequence].add(region)
        region.sequence = sequence

        """TODO
        if rebuild_table:
            self._rebuildListing()
        """
        if select:
            self._toggle_active(region, select_on_structures=not session_restore)
        if assoc_with:
            try:
                self.associated_regions[assoc_with].append(region)
            except KeyError:
                self.associated_regions[assoc_with] = [region]
            region.associated_with = assoc_with
        return region

    def raiseRegion(self, region, rebuild_table=True):
        if not region:
            self.seq_canvas.mav.status("No active region",
                                color="red")
            return
        if not isinstance(region, Region):
            for r in region[::-1]:
                self.raiseRegion(r)
            return
        index = self.regions.index(region)
        if index == 0:
            return
        self.regions.remove(region)
        self.regions.insert(0, region)
        for lowerRegion in self.regions[1:]:
            if lowerRegion.blocks and lowerRegion.shown:
                region.raise_above(lowerRegion)
                break
        if rebuild_table:
            self._rebuildListing()

    def redrawRegions(self, justGapping=False, cullEmpty=False):
        for region in self.regions[:]:
            if justGapping and region.cover_gaps:
                continue
            region.redraw()
            if cullEmpty and not region.blocks \
            and region != self.get_region("ChimeraX selection"):
                region.destroy()
        self.seq_canvas.adjustScrolling()
        
    def region_residues(self, region=None):
        if not region:
            region = self._drag_region
            if not region:
                return []
        sel_residues = []
        for block in region.blocks:
            sel_residues.extend(self._residues_in_block(block))
        return sel_residues

    def renameRegion(self, region, name=None):
        if not region:
            self.seq_canvas.mav.status("No active region",
                                color="red")
            return
        if not isinstance(region, Region):
            for r in region:
                self.renameRegion(r)
            return
        if region == self.get_region("ChimeraX selection"):
            self.seq_canvas.mav.status(
                "Cannot rename ChimeraX selection region",
                color="red")
            return
        if name == "ChimeraX selection":
            self.seq_canvas.mav.status("Cannot rename region as '%s'"
                % "ChimeraX selection", color="red")
            return
        if name is not None:
            region.name = name
            self._rebuildListing()
            if region in self.rename_dialogs:
                self.rename_dialogs[region].destroy()
                del self.rename_dialogs[region]
            return
        if region not in self.rename_dialogs:
            self.rename_dialogs[region] = RenameDialog(self, region)
        self.rename_dialogs[region].enter()

    def seeRegion(self, region=None):
        if not region:
            region = self.cur_region()
            if not region:
                return
        if isinstance(region, basestring):
            region_name = region
            region = self.get_region(region_name)
            if not region:
                replyobj.error("No region named '%s'\n" %
                                region_name)
                return
        
        self.seq_canvas.seeBlocks(region.blocks)

    def selected(self):
        """Return a list of selected regions"""
        return self.regionListing.selected()

    def showChimeraSelection(self):
        selRegion = self.get_region("ChimeraX selection", create=1,
            fill=self.seq_canvas.mav.prefs[SEL_REGION_INTERIOR],
            outline=self.seq_canvas.mav.prefs[SEL_REGION_BORDER])
        selRegion.clear()

        resDict = {}
        for res in currentResidues():
            resDict[res] = 1
        blocks = []
        for aseq in self.seq_canvas.seqs:
            try:
                matchMaps = aseq.matchMaps
            except AttributeError:
                continue
            for matchMap in matchMaps.values():
                start = None
                end = None
                for i in range(len(aseq.ungapped())):
                    if i in matchMap \
                    and matchMap[i] in resDict:
                        if start is not None:
                            end = i
                        else:
                            end = start = i
                    else:
                        if start is not None:
                            blocks.append([aseq, 
                             aseq, aseq. \
                             ungapped2gapped(start
                             ), aseq. \
                             ungapped2gapped(end)])
                            start = end = None
                if start is not None:
                    blocks.append([aseq, aseq,
                        aseq.ungapped2gapped(start),
                        aseq.ungapped2gapped(end)])
        if blocks and self._firstChimeraShow:
            self._firstChimeraShow = False
            self.seq_canvas.mav.status(
                "ChimeraX selection region displayed.\n"
                "Preferences..Regions controls this display.\n")
        selRegion.add_blocks(blocks)
        self.raiseRegion(selRegion)

    def showPredictedSS(self, show):
        """show predicted secondary structure"""
        from gor import gorI
        from chimera.Sequence import defHelixColor, defStrandColor
        helixReg = self.get_region(self.PRED_HELICES_REG_NAME, create=show,
                            outline=defHelixColor)
        if not helixReg:
            return
        strandReg = self.get_region(self.PRED_STRANDS_REG_NAME, create=show,
                            outline=defStrandColor)
        helixReg.shown = show
        strandReg.shown = show
        if not show:
            return
        helixReg.clear(make_cb=False)  # callback will happen in 
        strandReg.clear(make_cb=False) # add_blocks below

        helices = []
        strands = []
        for aseq in self.seq_canvas.seqs:
            if hasattr(aseq, 'matchMaps') and aseq.matchMaps:
                # has real associated structure
                continue
            pred = gorI(aseq)
            inHelix = inStrand = 0
            for pos in range(len(pred)):
                ss = pred[pos]
                if pred[pos] == 'C':
                    inHelix = inStrand = 0
                    continue
                gapped = aseq.ungapped2gapped(pos)
                if ss == 'H':
                    inStrand = 0
                    if inHelix:
                        helices[-1][-1] = gapped
                    else:
                        helices.append([aseq, aseq,
                            gapped, gapped])
                        inHelix = 1
                else:
                    inHelix = 0
                    if inStrand:
                        strands[-1][-1] = gapped
                    else:
                        strands.append([aseq, aseq,
                            gapped, gapped])
                        inStrand = 1
        helixReg.add_blocks(helices)
        strandReg.add_blocks(strands)

    def showSeqRegions(self, seq=None):
        if seq not in self.sequence_regions:
            seq = None
        seqOrder = self._regMenuOrder(retVal="sequence")
        self.seqRegionMenu.invoke(seqOrder.index(seq))

    def showSS(self, show):
        """show actual secondary structure"""
        from chimera.Sequence import defHelixColor, defStrandColor
        helixReg = self.get_region(self.ACTUAL_HELICES_REG_NAME, create=show,
                fill=(1.0, 1.0, 0.8), outline=defHelixColor)
        strandReg = self.get_region(self.ACTUAL_STRANDS_REG_NAME, create=show,
                fill=(0.8, 1.0, 0.8), outline=defStrandColor)
        if helixReg:
            helixReg.shown = show
        if strandReg:
            strandReg.shown = show
        if not show:
            return
        helixReg.clear(make_cb=False)  # callback will happen in
        strandReg.clear(make_cb=False) # add_blocks below

        assocSeqs = {}
        helices = []
        strands = []
        for aseq in self.seq_canvas.mav.associations.values():
            assocSeqs[aseq] = 1
        for aseq in assocSeqs.keys():
            inHelix = inStrand = 0
            for pos in range(len(aseq.ungapped())):
                isHelix = isStrand = 0
                for matchMap in aseq.matchMaps.values():
                    try:
                        res = matchMap[pos]
                    except KeyError:
                        continue
                    if res.isHelix:
                        isHelix = 1
                    elif res.isStrand:
                        isStrand = 1
                gapped = aseq.ungapped2gapped(pos)
                if isHelix:
                    if inHelix:
                        helices[-1][-1] = gapped
                    else:
                        helices.append([aseq, aseq,
                                gapped, gapped])
                        inHelix = 1
                else:
                    if inHelix:
                        inHelix = 0
                if isStrand:
                    if inStrand:
                        strands[-1][-1] = gapped
                    else:
                        strands.append([aseq, aseq,
                                gapped, gapped])
                        inStrand = 1
                else:
                    if inStrand:
                        inStrand = 0
        helixReg.add_blocks(helices)
        strandReg.add_blocks(strands)

    def unmap(self, *args):
        if self._mod_assoc_handler_id:
            self.seq_canvas.mav.triggers.deleteHandler(MOD_ASSOC,
                            self._mod_assoc_handler_id)
        if self._motion_stop_handler_id:
            chimera.triggers.deleteHandler(MOTION_STOP,
                            self._motion_stop_handler_id)
        self._mod_assoc_handler_id = self._motion_stop_handler_id = None

    def update_table_cell(self, region, column, **kw):
        if region in self.regionListing.data:
            self.regionListing.updateCellWidget(region, column, **kw)

    def _clear_drag(self):
        if self._drag_lines:
            for box in self._drag_lines:
                for line in box:
                    self.seq_canvas.main_scene.removeItem(line)
            self._drag_lines = []
            self._bboxes = []
            if self._drag_region:
                self._drag_region.remove_last_block()

    def _column_pick(self, event):
        pos = event.scenePos()
        canvas_x, canvas_y = pos.x(), pos.y()
        block = self.seq_canvas.bounded_by(canvas_x, canvas_y, canvas_x, canvas_y)
        if block[0] is None or block[0] in self.seq_canvas.alignment.seqs:
            return None
        return block[2]

    def _cover_gapsCB(self, opt):
        for r in self.selected():
            r.set_cover_gaps(opt.get())

    def _delAssocCB(self, trigName, myData, delMatchMaps):
        for matchMap in delMatchMaps:
            key = (matchMap['mseq'], matchMap['aseq'])
            if key in self.associated_regions:
                self.deleteRegion(self.associated_regions[key][:])
                
    def _focus_cb(self, event, pref=None):
        if pref == "residue":
            funcs = [self._residueCB, self._regionResiduesCB]
        else:
            funcs = [self._regionResiduesCB, self._residueCB]

        for func in funcs:
            residues = func(event)
            if residues is None: # no residue/region 
                continue
            if not residues: # region with no structure residues
                return
            break
        if residues is None:
            return
        from Midas import cofr, window
        from chimera.selection import ItemizedSelection
        sel = ItemizedSelection()
        sel.add(residues)
        window(sel)
        cofr(sel)

    def _get_rgba(self, specified=None):
        if isinstance(specified, str):
            from chimerax.core.colors import BuiltinColors
            return BuiltinColors[specified].rgba
        return specified
    
    def _listingCB(self, val=None):
        regions = self.selected()
        self.cover_gapsOption.display(regions)
        if val is not None: # not during table rebuild
            for region in regions:
                if self.showVar.get():
                    region.shown = True
                if self.activateVar.get():
                    region.active = True
                if self.hideVar.get():
                    for hr in self.regions:
                        if hr not in regions \
                        and hr.name and not (
                        hr.name.startswith(self.seq_canvas.mav.GAP_REG_NAME_START)
                        or hr == self.get_region("ChimeraX selection")):
                            hr.shown = False

    def _mouse_down_cb(self, event):
        from PyQt5.QtCore import Qt
        if event.button() == Qt.RightButton:
            if event.modifiers() & Qt.ShiftModifier:
                self._focus_cb(event, pref="region")
            else:
                self._focus_cb(event, pref="residue")
            return
        pos = event.scenePos()
        self._start_x, self._start_y = pos.x(), pos.y()
        canvas = self.seq_canvas.main_view
        """TODO
        self._canvasHeight = canvas.winfo_height()
        self._canvasWidth = canvas.winfo_width()
        """
        self._clear_drag()

        self._drag_region = None
        if event.modifiers() & Qt.ShiftModifier:
            self._drag_region = self.cur_region()
        else:
            self._drag_region = None

    def _mouse_drag_cb(self, event=None):
        if not hasattr(self, '_start_x') or self._start_x is None:
            return
        """
        canvas = self.seq_canvas.main_view
        if not event:
            # callback from over-edge mouse drag
            controlDown = 0
            if self._dragX < 0:
                xscroll = int(self._dragX/14) - 1
                x = 0
            elif self._dragX >= self._canvasWidth:
                xscroll = int((self._dragX - self._canvasWidth)
                    / 14) + 1
                x = self._canvasWidth
            else:
                xscroll = 0
                x = self._dragX
            if self._dragY < 0:
                yscroll = int(self._dragY/14) - 1
                y = 0
            elif self._dragY >= self._canvasHeight:
                yscroll = int((self._dragY - self._canvasHeight)
                    / 14) + 1
                y = self._canvasHeight
            else:
                yscroll = 0
                y = self._dragY
            if xscroll:
                canvas.xview_scroll(xscroll, 'units')
            if yscroll:
                canvas.yview_scroll(yscroll, 'units')
                if self.seq_canvas.labelCanvas != canvas:
                    self.seq_canvas.labelCanvas.yview_scroll(yscroll, 'units')
            if xscroll or yscroll:
                self._after_id = canvas.after(100,
                            self._mouse_drag_cb)
            else:
                self._after_id = None
        else:
            x = event.x
            y = event.y
            controlDown = event.state & 4 == 4
            if x < 0 or x >= self._canvasWidth \
            or y < 0 or y >= self._canvasHeight:
                # should scroll
                self._dragX = x
                self._dragY = y
                if self._after_id:
                    # already waiting on a scroll
                    return
                self._after_id = canvas.after(100,
                            self._mouse_drag_cb)
                return
            else:
                # should not scroll
                if self._after_id:
                    canvas.after_cancel(self._after_id)
                    self._after_id = None
        """

        from PyQt5.QtCore import Qt
        control_down = event.modifiers() & Qt.ControlModifier
        pos = event.scenePos()
        canvas_x, canvas_y = pos.x(), pos.y()
        if abs(canvas_x - self._start_x) > 1 or abs(canvas_y - self._start_y) > 1:
            block = self.seq_canvas.bounded_by(canvas_x, canvas_y, self._start_x, self._start_y)
            if block[0] is None:
                self._clear_drag()
                return
            if not self._drag_region:
                rebuild_table = True
                """TODO
                rebuild_table = self.seqRegionMenu.index(Pmw.SELECT) == 0
                """
                settings = self.seq_canvas.mav.settings
                self._drag_region = self.new_region(blocks=[block], select=True,
                    outline=settings.new_region_border, fill=settings.new_region_interior,
                    cover_gaps=True, rebuild_table=rebuild_table)
                if not control_down and self._prev_drag:
                    self._prev_drag.destroy(rebuild_table=rebuild_table)
                    self._prev_drag = None
            elif not self._drag_lines:
                self._drag_region.add_block(block)
            else:
                self._drag_region.update_last_block(block)

            bboxes = []
            for block in self._drag_region.blocks:
                bboxes.extend(self.seq_canvas.bbox_list(cover_gaps=True, *block))
            for i in range(len(bboxes)):
                cur_bbox = bboxes[i]
                try:
                    prev_bbox = self._bboxes[i]
                except IndexError:
                    prev_bbox = None
                if cur_bbox == prev_bbox:
                    continue
                ul_x, ul_y, lr_x, lr_y = cur_bbox
                ul_x -= 1
                ul_y -= 1
                lr_x += 1
                lr_y += 1
                if not prev_bbox:
                    create_line = self.seq_canvas.main_scene.addLine
                    from PyQt5.QtGui import QPen
                    pen = QPen(Qt.DotLine)
                    drag_lines = []
                    drag_lines.append(create_line(ul_x, ul_y, ul_x, lr_y, pen))
                    drag_lines.append(create_line(ul_x, lr_y, lr_x, lr_y, pen))
                    drag_lines.append(create_line(lr_x, lr_y, lr_x, ul_y, pen))
                    drag_lines.append(create_line(lr_x, ul_y, ul_x, ul_y, pen))
                    self._drag_lines.append(drag_lines)
                else:
                    drag_lines = self._drag_lines[i]
                    drag_lines[0].setLine(ul_x, ul_y, ul_x, lr_y)
                    drag_lines[1].setLine(ul_x, lr_y, lr_x, lr_y)
                    drag_lines[2].setLine(lr_x, lr_y, lr_x, ul_y)
                    drag_lines[3].setLine(lr_x, ul_y, ul_x, ul_y)
            for i in range(len(bboxes), len(self._bboxes)):
                rm  = self.seq_canvas.main_scene.removeItem
                for line in self._drag_lines[i]:
                    rm(line)
            self._drag_lines = self._drag_lines[:len(bboxes)]
            self._bboxes = bboxes
        else:
            self._clear_drag()

    def _mouse_up_cb(self, event, double=False):
        canvas = self.seq_canvas.main_view
        if not self._drag_region:
            # maybe a region pick
            region = self._region(event)
            if region:
                if double:
                    self.enter()
                else:
                    self._toggle_active(region)
            else:
                # maybe a column pick
                col = self._column_pick(event)
                if col is not None:
                    residues = self._residues_in_block((
                        self.seq_canvas.seqs[0], self.seq_canvas.seqs[-1], col, col))
                    sel = ItemizedSelection()
                    sel.add(residues)
                    selectionOperation(sel)
        else:
            self._select_on_structures()
            self._prev_drag = self._drag_region
            rmsd = self._drag_region.rmsd
            if rmsd == None:
                self.seq_canvas.mav.status(
                    "Shift-drag to add to region; "
                    "control-drag to add new region\n"
                    "Tools->Region Browser to change colors; "
                    "control left/right arrow to realign region",
                    blankAfter=120)
            else:
                mav = self.seq_canvas.mav
                mav.status("Region RMSD: %.3f" % rmsd)
                mav.session.logger.info("%s RMSD: %.3f\n" % (self._drag_region, rmsd))
        self._start_x, self._start_y = None, None
        if self._after_id:
            canvas.after_cancel(self._after_id)
            self._after_id = None
        self._drag_region = None
        self._clear_drag()

    def _preAddLines(self, lines):
        for region in self.regions:
            region._addLines(lines)

    def _pre_remove_lines(self, lines):
        for seq, regions in self.sequence_regions.items():
            if seq in lines:
                for region in list(regions):
                    self.deleteRegion(region, rebuild_table=False)

        for region in self.regions:
            region._del_lines(lines)

    def _rebuildListing(self, *args):
        index = self.seqRegionMenu.index(Pmw.SELECT)
        regionSet = self.sequence_regions[self._regMenuOrder(retVal="sequence")[index]]
        regions = []
        for region in self.regions:
            if region in regionSet:
                regions.append(region)
        # filter based on source
        sources = set([r.source for r in regions])
        for source in sources:
            if source not in self.sourceControls:
                var = Tkinter.IntVar(self.uiMaster())
                var.set(True)
                but = Tkinter.Checkbutton(self.sourceButtonArea, text=source,
                    variable=var, command=self._rebuildListing)
                self.sourceControls[source] = (var, but)
        # if multiple sources, or if the single source is set to be hidden,
        # show the controls (and filter the regions)
        if sources and (len(sources) > 1
                or not self.sourceControls[list(sources)[0]][0].get()):
            if None in sources:
                order = [None]
                sources.remove(None)
            else:
                order = []
            order += sorted(list(sources))
            # hide all first
            for var, but in self.sourceControls.values():
                but.grid_forget()
            for i, src in enumerate(order):
                var, but = self.sourceControls[src]
                but.grid(row=0, column=i, padx="0.1i")
                if not var.get():
                    regions = [r for r in regions if r.source != src]
            self.sourceArea.grid()
        else:
            self.sourceArea.grid_remove()
        self.regionListing.setData(regions)
        self.regionListing.columnUpdate(self.rmsdCol, display=index==0, immediateRefresh=False)
        self.regionListing.columnUpdate(self.startCol, display=index>0, immediateRefresh=False)
        self.regionListing.columnUpdate(self.endCol, display=index>0)

    def _region(self, event):
        pos = event.scenePos()
        x, y = pos.x(), pos.y()
        for region in self.regions:
            if region.contains(x, y):
                return region
        return None

    def _region_destroyed_cb(self, region, rebuild_table=True):
        if region == self._cur_region:
            self._toggle_active(region)
        self.regions.remove(region)
        """TODO
        if rebuild_table:
            self._rebuildListing()
        """
        if region == self._prev_drag:
            self._prev_drag = None
        if region == self._drag_region:
            self._drag_region = None
        if region == self._highlighted_region:
            self._highlighted_region = None
        if region in self.rename_dialogs:
            self.rename_dialogs[region].destroy()
            del self.rename_dialogs[region]

    def _regionResiduesCB(self, event):
        region = self._region(event)
        if not region:
            return None
        residues = []
        for block in region.blocks:
            residues.extend(self._residues_in_block(block))
        return residues

    def _region_size_changed_cb(self, region):
        """TODO
        if region.name is None and self.seqRegionMenu.index(Pmw.SELECT) == 0:
            self._rebuildListing()
        """

    def _regMenuOrder(self, retVal="text"):
        seqs = self.sequence_regions.keys()
        seqs.remove(None)
        seqs.sort(lambda s1, s2, seqs=self.seq_canvas.seqs:
                cmp((s1.name, seqs.index(s1)), (s2.name, seqs.index(s2))))
        if retVal == "text":
            return [self.seqRegionMenu.component('menu').entrycget(0, 'label')] + \
                [seq.name for seq in seqs]
        else:
            return [None] + seqs

    def _residueCB(self, event):
        canvas = self.seq_canvas.main_view
        canvasX = canvas.canvasx(event.x)
        canvasY = canvas.canvasy(event.y)
        block = self.seq_canvas.bounded_by(canvasX, canvasY,
                            canvasX, canvasY)
        if block[0] is None:
            return None
        return self._residues_in_block(block)

    def _residues_in_block(self, block):
        line1, line2, pos1, pos2 = block

        residues = []
        seqs = self.seq_canvas.alignment.seqs
        try:
            index1 = seqs.index(line1)
        except ValueError:
            index1 = 0
        try:
            index2 = seqs.index(line2) + 1
        except ValueError:
            index2 = 0
        for aseq in seqs[index1:index2]:
            try:
                match_maps = aseq.match_maps
            except AttributeError:
                continue
            for match_map in match_maps.values():
                for gapped in range(pos1, pos2+1):
                    ungapped = aseq.gapped_to_ungapped(gapped)
                    if ungapped is None:
                        continue
                    try:
                        res = match_map[ungapped]
                    except KeyError:
                        continue
                    residues.append(res)
        return residues

    def _select_on_structures(self, region=None):
        # highlight on chimerax structures
        self._sel_change_from_self = True
        self.seq_canvas.mav.session.selection.clear()
        from chimerax.core.atomic import Residues
        Residues(self.region_residues(region)).atoms.selected = True
        self._sel_change_from_self = False

    def _selChangeCB(self, trigName, myData, trigData):
        selRegion = self.get_region("ChimeraX selection", create=1,
            fill=self.seq_canvas.mav.prefs[SEL_REGION_INTERIOR],
            outline=self.seq_canvas.mav.prefs[SEL_REGION_BORDER])
        if self._sel_change_from_self:
            selRegion.clear()
        else:
            self.showChimeraSelection()

    def _seq_renamed_cb(self, _1, trig_data):
        seq, old_name = trig_data
        if seq not in self.sequence_regions:
            return
        """
        prevItem = self.seqRegionMenu.getvalue()
        if prevItem == old_name:
            newItem = seq.name
        else:
            newItem = prevItem
        self.seqRegionMenu.setitems(self._regMenuOrder(), index=newItem)
        """

    def _showSelCB(self):
        # also called from PrefDialog.py
        if self.seq_canvas.mav.prefs[SHOW_SEL]:
            self.showChimeraSelection()
            self._selChangeHandler = chimera.triggers.addHandler(
                "selection changed", self._selChangeCB, None)
        else:
            chimera.triggers.deleteHandler("selection changed",
                        self._selChangeHandler)
            self._selChangeHandler = None
            selRegion = self.get_region("ChimeraX selection")
            if selRegion:
                selRegion.destroy()

    def _toggle_active(self, region, select_on_structures=True):
        if self._cur_region is not None and self._cur_region == region:
            region.dehighlight()
            self._cur_region = None
        else:
            self._cur_region = region
            self.highlight(region, select_on_structures=select_on_structures)

"""
from OpenSave import OpenModeless
class ScfDialog(OpenModeless):
    title = "Load SCF/Seqsel File"

    def __init__(self, colorStructuresDefault, **kw):
        kw['filters'] = [("SCF", ["*.scf", "*.seqsel"])]
        kw['defaultFilter'] = 0
        kw['clientPos'] = 's'
        self.colorStructuresDefault = colorStructuresDefault
        OpenModeless.__init__(self, **kw)

    def fillInUI(self, parent):
        OpenModeless.fillInUI(self, parent)
        self.colorStructureVar = Tkinter.IntVar(self.clientArea)
        self.colorStructureVar.set(self.colorStructuresDefault)

        Tkinter.Checkbutton(self.clientArea,
                text="Color structures also",
                variable=self.colorStructureVar).grid()

class RenameDialog(ModelessDialog):
    buttons = ('OK', 'Cancel')
    default = 'OK'

    def __init__(self, region_browser, region):
        self.title = "Rename '%s' Region" % region_name(region,
                region_browser.seq_canvas.mav.prefs)
        self.region_browser = region_browser
        self.region = region
        ModelessDialog.__init__(self)

    def map(self, e=None):
        self.renameOpt._option.focus_set()

    def fillInUI(self, parent):
        from chimera.tkoptions import StringOption
        self.renameOpt = StringOption(parent, 0, "Rename region to",
                                "", None)
    def Apply(self):
        newName = self.renameOpt.get().strip()
        if not newName:
            self.enter()
            from chimera import UserError
            raise UserError("Must supply a new region name or "
                            "click Cancel")
        self.region_browser.renameRegion(self.region, newName)

    def destroy(self):
        self.region = None
        self.region_browser = None
        ModelessDialog.destroy(self)
        
from SeqCanvas import ellipsisName
def region_name(region, prefs):
    if region.name:
        return unicode(region)
    return ellipsisName(unicode(region), prefs[REGION_NAME_ELLIPSIS])
"""
