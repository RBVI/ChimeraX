# vim: set expandtab ts=4 sw=4:

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
    def __init__(self, region_manager, name=None, init_blocks=[], shown=True,
            border_rgba=None, interior_rgba=None, name_prefix="",
            cover_gaps=False, source=None, read_only=False):
        self._name = name
        self.name_prefix = name_prefix
        self.region_manager = region_manager
        self.seq_canvas = self.region_manager.seq_canvas
        self.scene = self.seq_canvas.main_scene
        self._border_rgba = self._prev_border_rgba = border_rgba
        self._interior_rgba = self._prev_interior_rgba = interior_rgba
        self.cover_gaps = cover_gaps
        self.source = source
        self.read_only = read_only
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
        self.region_manager._toggle_active(self)

    def get_active(self):
        return self.highlighted

    active = property(get_active, set_active)

    def add_block(self, block):
        self.add_blocks([block])

    def add_blocks(self, block_list, make_cb=True):
        self.blocks.extend(block_list)
        if make_cb:
            self.region_manager._region_size_changed_cb(self)
        if self._name is None:
            self._notify_tool('name')
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
                self._items.append(self.scene.addRect(x1, y1, x2-x1, y2-y1, **kw))
                if self.name:
                    self._items[-1].setToolTip(str(self))
                if len(self._items) > 1:
                    self._items[-1].setZValue(self._items[-2].zValue()-0.001)
                    continue

                regions = self.region_manager.regions
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
        import numpy
        if numpy.array_equal(self.border_rgba, rgba):
            return
        self._prev_border_rgba = self._border_rgba
        self._border_rgba = rgba
        # kind of complicated due to highlighting; just redraw
        self.redraw()

    border_rgba = property(get_border_rgba, set_border_rgba)

    @property
    def prev_border_rgba(self):
        return self._prev_border_rgba

    def clear(self, make_cb=True):
        if self.blocks:
            self.blocks = []
            for item in self._items:
                self.scene.removeItem(item)
            self._items = []
            if make_cb:
                self.region_manager._region_size_changed_cb(self)

    def contains(self, x, y):
        from Qt.QtCore import QPointF
        for item in self.scene.items(QPointF(x, y)):
            if item in self._items:
                return True
        return False

    def dehighlight(self):
        if not self.highlighted:
            return
        self.highlighted = False
        self.redraw()
        self._notify_tool('active')

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

    def _destroy(self, rebuild_table=True):
        # In most circumstances, use region_manager.delete_region() instead
        for item in self._items:
            self.scene.removeItem(item)
        self.blocks = []
        self.sequence = self.associated_with = None
        self.region_manager._region_destroyed_cb(self, rebuild_table=rebuild_table)

    def _disp_border_rgba(self):
        # account for highlighting
        if self.border_rgba is None and self.highlighted:
            return (0.0, 0.0, 0.0, 1.0)
        return self.border_rgba

    @property
    def display_name(self):
        return str(self)
        # seemed like using ellipsis wasn't really necessary...
        """
        if self.name:
            return str(self)
        from .seq_canvas import ellipsis_name
        return ellipsis_name(str(self), self.region_manager.seq_canvas.sv.settings.region_name_ellipsis)
        """

    @display_name.setter
    def display_name(self, val):
        self.name = val

    def highlight(self):
        if self.highlighted:
            return
        self.highlighted = True
        self.redraw()
        self._notify_tool('active')

    def get_interior_rgba(self):
        return self._interior_rgba

    def set_interior_rgba(self, rgba):
        import numpy
        if numpy.array_equal(self.interior_rgba, rgba):
            return
        self._prev_interior_rgba = self._interior_rgba
        self._interior_rgba = rgba
        if not self._items:
            return
        brush = self._items[0].brush()
        from Qt.QtCore import Qt
        if rgba is not None:
            from Qt.QtGui import QColor
            brush.setStyle(Qt.SolidPattern)
            brush.setColor(QColor(*[int(x*255.0 + 0.5) for x in rgba]))
        else:
            brush.setStyle(Qt.NoBrush)
        for item in self._items:
            item.setBrush(brush)
        self.redraw()

    interior_rgba = property(get_interior_rgba, set_interior_rgba)

    @property
    def prev_interior_rgba(self):
        return self._prev_interior_rgba

    def lower_below(self, other_region):
        if not self._items or not other_region._items:
            return
        self._items[0].setZValue(other_region._items[0].zValue()-1)
        for i in range(1, len(self._items)):
            self._items[i].setZValue(self._items[i-1].zValue()-0.001)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        if val == self._name:
            return
        if self.read_only:
            self.region_manager.seq_canvas.sv.session.logger.error(
                "Cannot rename %s region" % self.name)
            return
        if val == None:
            for item in self._items:
                item.setToolTip("")
        self._name = val
        if val:
            for item in self._items:
                item.setToolTip(str(self))
        self._notify_tool('name')

    def _notify_tool(self, category):
        sv = self.region_manager.seq_canvas.sv
        sv._regions_tool_notification(category, self)

    def raise_above(self, other_region):
        if not self._items or not other_region._items:
            return
        self._items[0].setZValue(other_region._items[0].zValue()+1)
        for i in range(1, len(self._items)):
            self._items[i].setZValue(self._items[i-1].zValue()-0.001)

    def _rect_kw(self):
        kw = {}
        from Qt.QtGui import QBrush, QPen, QColor
        kw['pen'] = pen = QPen()
        kw['brush'] = brush = QBrush()
        from Qt.QtCore import Qt
        if self.interior_rgba is not None:
            brush.setColor(rgba_to_qcolor(self.interior_rgba))
            brush.setStyle(Qt.SolidPattern)
        else:
            brush.setStyle(Qt.NoBrush)
        if self._disp_border_rgba() is not None:
            pen.setColor(rgba_to_qcolor(self._disp_border_rgba()))
            if self.highlighted:
                pen.setStyle(Qt.DashLine)
            else:
                pen.setStyle(Qt.SolidLine)
        else:
            # no outline will not be filled so...
            if self.interior_rgba is None:
                pen.setStyle(Qt.NoPen)
            else:
                pen.setColor(rgba_to_qcolor(self.interior_rgba))
                pen.setStyle(Qt.SolidLine)
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
            self.region_manager.delete_region(self)
        elif make_cb:
            self.region_manager._region_size_changed_cb(self)
            if self._name is None:
                self._notify_tool('name')

    def restore_state(self, state):
        self._name = state['_name']
        self.name_prefix = state['name_prefix']
        self._border_rgba = state['_border_rgba']
        self._interior_rgba = state['_interior_rgba']
        self._prev_border_rgba = state.get('_prev_border_rgba', self._border_rgba)
        self._prev_interior_rgba = state.get('_prev_interior_rgba', self._interior_rgba)
        self.cover_gaps = state['cover_gaps']
        self.source = state['source']
        self.highlighted = state['highlighted']
        self._shown = state['_shown']
        self._active = state['_active']
        self.sequence = state['sequence']
        self.associated_with = state['associated_with']
        self.read_only = state.get('read_only', self._name == "ChimeraX selection")
        self.add_blocks(state['blocks'], make_cb=False)
        return state

    def get_rmsd(self):
        num_d = sum_d2 = 0
        rmsd_chains = set(self.seq_canvas.alignment.rmsd_chains)
        from chimerax.geometry import distance_squared
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
                for chain, match_map in seq.match_maps.items():
                    if chain in rmsd_chains:
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

    def state(self):
        state = {}
        state['_name'] = self._name
        state['name_prefix'] = self.name_prefix
        state['_border_rgba'] = self._border_rgba
        state['_interior_rgba'] = self._interior_rgba
        state['_prev_border_rgba'] = self._prev_border_rgba
        state['_prev_interior_rgba'] = self._prev_interior_rgba
        state['cover_gaps'] = self.cover_gaps
        state['source'] = self.source
        state['highlighted'] = self.highlighted
        state['_shown'] = self._shown
        state['_active'] = self._active
        state['sequence'] = self.sequence
        state['associated_with'] = self.associated_with
        state['blocks'] = self.blocks
        state['read_only'] = self.read_only
        return state

    def set_cover_gaps(self, cover):
        if cover != self.cover_gaps:
            self.cover_gaps = cover
            self.redraw()

    def set_shown(self, val):
        if bool(val) == self._shown:
            return
        self._shown = bool(val)
        self.redraw()
        self._notify_tool("shown")

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

class RegionManager:

    PRED_HELICES_REG_NAME = "predicted helices"
    PRED_STRANDS_REG_NAME = "predicted strands"
    PRED_SS_REG_NAMES = [PRED_HELICES_REG_NAME, PRED_STRANDS_REG_NAME]
    ACTUAL_HELICES_REG_NAME = "structure helices"
    ACTUAL_STRANDS_REG_NAME = "structure strands"
    ACTUAL_SS_REG_NAMES = [ACTUAL_HELICES_REG_NAME, ACTUAL_STRANDS_REG_NAME]
    SS_REG_NAMES = PRED_SS_REG_NAMES + ACTUAL_SS_REG_NAMES
    ENTIRE_ALIGNMENT_REGIONS = "entire alignment"

    def __init__(self, seq_canvas):
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
        self._mod_assoc_handler_id = self._motion_stop_handler_id = None
        self.sequence_regions = { None: set() }
        self._cur_region = None
        self._sel_change_handler = None
        seq_canvas.main_scene.keyPressEvent = self._key_press_cb
        settings = seq_canvas.sv.settings
        self._sel_change_from_self = False
        self._first_sel_region_show = True
        if settings.show_sel:
            self._show_sel_cb()


    def clear_regions(self, do_single_seq_regions=True):
        if do_single_seq_regions:
            for region in self.regions[:]:
                region._destroy()
            self.associated_regions.clear()
            self.sequence_regions = { None: set() }
        else:
            single_seq_regions = set()
            for seq, regions in self.sequence_regions.items():
                if seq is not None:
                    single_seq_regions.update(regions)
            for region in self.regions[:]:
                if region not in single_seq_regions:
                    self.delete_regions(region)

    def copy_region(self, region, name=None, **kw):
        if not region:
            self.seq_canvas.sv.status("No active region", color="red")
            return
        if not isinstance(region, Region):
            for r in region:
                self.copy_region(r)
            return
        if name is None:
            initial_name = "Copy of " + str(region)
        else:
            initial_name = name
        seq = region.sequence
        copy = self.new_region(name=initial_name,
                blocks=region.blocks, fill=region.interior_rgba,
                outline=region.border_rgba, sequence=seq,
                cover_gaps=region.cover_gaps, **kw)
        self.seq_canvas.sv._regions_tool_notification('select', copy)
        return copy

    def cur_region(self):
        return self._cur_region

    def delete_region(self, region, rebuild_table=True):
        if not region:
            self.seq_canvas.sv.status("No active region", color="red")
            return
        if not isinstance(region, Region):
            for r in region:
                self.delete_region(r, rebuild_table=(r == region[-1]))
            return
        if region.read_only:
            self.seq_canvas.sv.status("Cannot delete %s region" % region.name, color="red")
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
            region._destroy(rebuild_table=rebuild_table)
            if seq and not regions:
                del self.sequence_regions[seq]
                """
                self.seqRegionMenu.setitems(self._regMenuOrder())
                if rebuild_table:
                    self.seqRegionMenu.invoke(0)
                """
            if rebuild_table:
                self.seq_canvas.sv._regions_tool_notification('delete', region)

    def destroy(self):
        """
        self.regionListing.destroy()
        self.seq_canvas.sv.triggers.deleteHandler(DEL_ASSOC,
                            self._delAssocHandlerID)
        if self._mod_assoc_handler_id:
            self.seq_canvas.sv.triggers.deleteHandler(MOD_ASSOC,
                            self._mod_assoc_handler_id)
        if self._motion_stop_handler_id:
            chimera.triggers.deleteHandler(MOTION_STOP,
                            self._motion_stop_handler_id)
        self.seq_canvas.sv.triggers.deleteHandler(PRE_DEL_SEQS,
            self._pre_del_seqs_handler_id)
        self.seq_canvas.sv.triggers.deleteHandler(SEQ_RENAMED,
                            self._seqRenamedHandlerID)
        """
        if self._sel_change_handler:
            self._sel_change_handler.remove()

    """TODO: also change _mouse_up_cb handling of double-click to raise the region manager
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
        prefs = self.seq_canvas.sv.prefs
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
        self._first_sel_region_show = True
        if self.seq_canvas.sv.prefs[SHOW_SEL]:
            self._show_sel_cb()

        cb = lambda e, s=self: s.deleteRegion(s.selected())
        parent.winfo_toplevel().bind('<Delete>', cb)
        parent.winfo_toplevel().bind('<BackSpace>', cb)

        self._delAssocHandlerID = self.seq_canvas.sv.triggers.addHandler(
                    DEL_ASSOC, self._delAssocCB, None)
    """

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

    """
    def lastDrag(self):
        return self._prev_drag
    """

    def load_scf_file(self, path, color_structures=None):
        if path is None:
            from chimerax.ui.open_save import OpenDialog
            dlg = OpenDialog(self.seq_canvas.main_view, caption="Load Sequence Coloring File")
            dlg.setNameFilter("SCF files (*.scf *.seqsel)")
            from Qt.QtWidgets import QCheckBox
            cbox = QCheckBox("Also color associated structures")
            sv = self.seq_canvas.sv
            settings = sv.settings
            cbox.setChecked(settings.scf_colors_structures)
            from Qt.QtWidgets import QHBoxLayout
            layout = QHBoxLayout()
            layout.addWidget(cbox)
            dlg.custom_area.setLayout(layout)
            path = dlg.get_path()
            if path is None:
                return
            settings.scf_colors_structures = cbox.isChecked()
            from chimerax.core.commands import quote_path_if_necessary as q_if, run
            from . import subcommand_name
            run(self.seq_canvas.sv.session, "sequence %s %s scfLoad %s color %s"
                % (subcommand_name, q_if(sv.alignment.ident),
                q_if(path), settings.scf_colors_structures))
            return

        if color_structures is None:
            color_structures = self.settings.scf_colors_structures

        seqs = self.seq_canvas.alignment.seqs
        from chimerax.io import open_input
        scf_file = open_input(path, 'utf-8')
        line_num = 0
        region_info = {}
        from chimerax.core.errors import UserError
        for line in scf_file.readlines():
            line_num += 1
            line.strip()
            if not line or line[0] == '#' or line.startswith('//'):
                continue
            for comment_intro in ['//', '#']:
                comment_pos = line.find(comment_intro)
                if comment_pos >= 0:
                    break
            comment = None
            if comment_pos >= 0:
                comment = line[comment_pos + len(comment_intro):].strip()
                line = line[:comment_pos].strip()
            if not comment:
                comment = "SCF region"

            try:
                pos1, pos2, seq1, seq2, r, g, b = [int(x) for x in line.split()]
                if seq1 == -1:
                    # internal to jevtrace/webmol
                    continue
                if seq1 == 0:
                    seq2 = -1
                else:
                    seq1 -= 1
                    seq2 -= 1
            except Exception:
                try:
                    pos, seq, r, g, b = [int(x) for x in line.split()]
                    pos1 = pos2 = pos
                except Exception:
                    scf_file.close()
                    raise UserError("Bad format for line %d of %s [not 5 or 7 integers]"
                        % (line_num, path))
                if seq == 0:
                    seq1 = 0
                    seq2 = -1
                else:
                    seq1 = seq2 = seq - 1
            key = ((r, g, b), comment)
            if key in region_info:
                region_info[key].append((seqs[seq1],
                    seqs[seq2], pos1, pos2))
            else:
                region_info[key] = [(seqs[seq1], seqs[seq2], pos1, pos2)]
        scf_file.close()

        if not region_info:
            raise UserError("No annotations found in %s" % path)
        if isinstance(path, str):
            import os.path
            source = os.path.basename(path)
        else:
            source = "SCF data"
        for rbg_comment, blocks in region_info.items():
            rgb, comment = rbg_comment
            region = self.new_region(source=source,
                blocks=blocks, name=comment, fill=[c/255.0 for c in rgb], cover_gaps=True)
            if not color_structures:
                continue
            rgba = list(rgb) + [255]
            for res in self.region_residues(region):
                res.ribbon_color = rgba
                for a in res.atoms:
                    a.color = rgba
        self.seq_canvas.sv.status("%d scf regions created" % len(region_info))

    def lower_region(self, region, rebuild_table=True):
        if not region:
            self.seq_canvas.sv.status("No active region", color="red")
            return
        if not isinstance(region, Region):
            for r in region[::-1]:
                self.lower_region(r)
            return
        index = self.regions.index(region)
        if index == len(self.regions) - 1:
            return
        self.regions.remove(region)
        self.regions.append(region)
        for higher_region in self.regions[-2::-1]:
            if higher_region.blocks and higher_region.shown:
                region.lower_below(higher_region)
                break
        if rebuild_table:
            self.seq_canvas.sv._regions_tool_notification('lower', region)

    """
    def map(self, *args):
        refreshRMSD = lambda *args: self.regionListing.refresh()
        self._mod_assoc_handler_id = self.seq_canvas.sv.triggers.addHandler(
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
    """

    def new_region(self, name=None, blocks=[], fill=None, outline=None,
            name_prefix="", select=False, assoc_with=None, cover_gaps=True,
            after="ChimeraX selection", rebuild_table=True,
            session_restore=False, sequence=None, **kw):
        if not name and not name_prefix:
            # possibly first user-dragged region
            for reg in self.regions:
                if not reg.name and not reg.name_prefix:
                    # another user region
                    break
            else:
                self.seq_canvas.sv.status("Use delete/backspace key to remove regions")
        interior = get_rgba(fill)
        border = get_rgba(outline)
        clipped_blocks = []
        warn = self.seq_canvas.sv.session.logger.warning
        seqs = self.seq_canvas.alignment.seqs
        target = "sequence" if len(seqs) == 1 else "alignment"
        for seq1, seq2, start, end in blocks:
            if start < 0:
                warn("Region %s starts before start of %s; truncating"
                    % (("dragged region" if name is None else name), target))
                start = 0
            if end >= len(seqs[0]):
                warn("Region '%s' extends past end of %s; truncating"
                    % (("(dragged)" if name is None else name), target))
                end = len(seqs[0]) - 1
            clipped_blocks.append((seq1, seq2, start, end))
        region = Region(self, init_blocks=clipped_blocks, name=name, name_prefix=name_prefix,
                border_rgba=border, interior_rgba=interior, cover_gaps=cover_gaps, **kw)
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

        if select:
            self._toggle_active(region, select_on_structures=not session_restore)
        if assoc_with:
            try:
                self.associated_regions[assoc_with].append(region)
            except KeyError:
                self.associated_regions[assoc_with] = [region]
            region.associated_with = assoc_with
        if rebuild_table:
            self.seq_canvas.sv._regions_tool_notification('new', region)
        return region

    def raise_region(self, region, rebuild_table=True):
        if not region:
            self.seq_canvas.sv.status("No active region", color="red")
            return
        if not isinstance(region, Region):
            for r in region[::-1]:
                self.raise_region(r)
            return
        index = self.regions.index(region)
        if index == 0:
            return
        self.regions.remove(region)
        self.regions.insert(0, region)
        for lower_region in self.regions[1:]:
            if lower_region.blocks and lower_region.shown:
                region.raise_above(lower_region)
                break
        if rebuild_table:
            self.seq_canvas.sv._regions_tool_notification('raise', region)

    def redraw_regions(self, just_gapping=False, cull_empty=False):
        for region in self.regions[:]:
            if just_gapping and region.cover_gaps:
                continue
            region.redraw()
            if cull_empty and not region.blocks \
            and region != self.get_region("ChimeraX selection"):
                self.delete_region(region)

    def region_info(self, region):
        if not region:
            self.seq_canvas.sv.status("No active region", color="red")
            return
        if not isinstance(region, Region):
            for r in region:
                self.region_info(r)
            return

        from html import escape
        from chimerax.core.logger import html_table_params
        lines = [
            '<table %s>' % html_table_params,
            '  <thead>',
            '    <tr>',
            '      <th colspan="2">Position Coverage for &quot;%s&quot;</th>' % escape(str(region)),
            '    </tr>',
            '  </thead>',
            '  <tbody>',
        ]

        # gapped
        if region.blocks:
            coverage_value =  ",".join(["%d-%d" % (b[-2]+1, b[-1]+1) for b in region.blocks])
        else:
            coverage_value = "empty region"
        lines.extend([
            '    <tr>',
            '      <td style="text-align:center">Alignment</td>',
            '      <td style="text-align:center">%s</td>' % coverage_value,
        ])

        # ungapped
        seqs = self.seq_canvas.alignment.seqs
        chain_info = {}
        for i, seq in enumerate(seqs):
            blocks = []
            for seq1, seq2, pos1, pos2 in region.blocks:
                if i < seqs.index(seq1) or i > seqs.index(seq2):
                    continue
                # find edges that aren't in gaps
                for p1 in range(pos1, pos2+1):
                    if not seq.is_gap_character(seq[p1]):
                        break
                else: # all gap
                    continue
                for p2 in range(pos2, pos1-1, -1):
                    if not seq.is_gap_character(seq[p2]):
                        break
                u1, u2 = [seq.gapped_to_ungapped(p) for p in (p1, p2)]
                blocks.append((u1, u2))
                for chain, mmap in seq.match_maps.items():
                    start = None
                    for u in range(u1, u2+1):
                        try:
                            r = mmap[u]
                        except KeyError:
                            continue
                        start = r
                        break
                    if start is None:
                        continue
                    for u in range(u2, u1-1, -1):
                        try:
                            r = mmap[u]
                        except KeyError:
                            continue
                        end = r
                        break
                    chain_info.setdefault(chain, []).append((start, end))
            if blocks:
                off = 1 if seq.numbering_start is None else seq.numbering_start
                lines.extend([
                    '    <tr>',
                    '      <td style="text-align:center">%s</td>' % seq.name,
                    '      <td style="text-align:center">%s</td>' % ",".join(
                        ["%d-%d" % (u1+off, u2+off) for u1, u2 in blocks]),
                ])

        # asspciated structures
        if chain_info:
            rstr = lambda r: r.string(omit_structure=True, omit_chain=True)
            for chain in sorted(list(chain_info.keys())):
                ranges = []
                for start, end in chain_info[chain]:
                    if start == end:
                        ranges.append(rstr(start))
                    else:
                        ranges.append(rstr(start) + '-' + rstr(end))
                lines.extend([
                    '    <tr>',
                    '      <td style="text-align:center">%s</td>' % chain,
                    '      <td style="text-align:center">%s</td>' % ",".join(ranges),
                ])
        else:
            lines.extend([
                '    <tr>',
                '      <td style="text-align:center" colspan="2">Region has no associated structure</td>',
                '    </tr>',
            ])

        lines.extend([
            '  </tbody>',
            '</table>',
        ])
        self.seq_canvas.sv.session.logger.info('\n'.join(lines), is_html=True)
        self.seq_canvas.sv.status("Region info reported in log")

    def region_residues(self, region=None):
        if not region:
            region = self._drag_region
            if not region:
                return []
        sel_residues = []
        for block in region.blocks:
            sel_residues.extend(self._residues_in_block(block))
        return sel_residues

    def restore_state(self, state):
        self.clear_regions()
        for region_state in state['regions']:
            r = Region(self)
            self.regions.append(r)
            r.restore_state(region_state)
        hr = state['_highlighted_region']
        self._highlighted_region = None if hr is None else self.regions[hr]
        self.associated_regions = { k: [self.regions[ri] for ri in v]
            for k,v in state['associated_regions'].items() }
        self.sequence_regions = { k: set([ self.regions[ri] for ri in v ])
            for k,v in state['sequence_regions'].items() }
        cr = state['_cur_region']
        self._cur_region = None if cr is None else self.regions[cr]
        dr = state['_drag_region']
        self._drag_region = None if dr is None else self.regions[dr]
        pd = state['_prev_drag']
        self._prev_drag = None if pd is None else self.regions[pd]
        return state

    def state(self):
        state = {}
        region_state = state['regions'] = []
        for region in self.regions:
            region_state.append(region.state())
        state['_highlighted_region'] = None if self._highlighted_region is None \
            else self.regions.index(self._highlighted_region)
        state['associated_regions'] = { k: [ self.regions.index(r) for r in v ]
            for k,v in self.associated_regions.items() }
        state['sequence_regions'] = { k: [ self.regions.index(r) for r in v ]
            for k,v in self.sequence_regions.items() }
        state['_cur_region'] = None if self._cur_region is None \
            else self.regions.index(self._cur_region)
        state['_drag_region'] = None if self._drag_region is None \
            else self.regions.index(self._drag_region)
        state['_prev_drag'] = None if self._prev_drag is None \
            else self.regions.index(self._prev_drag)
        return state

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
        if self._cur_region:
            return [self._cur_region]
        return []
        """TODO
        return self.regionListing.selected()
        """

    def show_chimerax_selection(self):
        sv = self.seq_canvas.sv
        sel_region = self.get_region("ChimeraX selection", create=True, read_only=True,
            fill=sv.settings.sel_region_interior, outline=sv.settings.sel_region_border)
        sel_region.clear()

        from chimerax.atomic import selected_residues
        sel_residues = set(selected_residues(self.seq_canvas.sv.session))
        blocks = []
        for aseq in self.seq_canvas.alignment.seqs:
            for match_map in aseq.match_maps.values():
                start = None
                end = None
                for i in range(len(aseq.ungapped())):
                    if i in match_map and match_map[i] in sel_residues:
                        if start is not None:
                            end = i
                        else:
                            end = start = i
                    else:
                        if start is not None:
                            blocks.append([aseq, aseq, aseq.ungapped_to_gapped(start),
                                aseq.ungapped_to_gapped(end)])
                            start = end = None
                if start is not None:
                    blocks.append([aseq, aseq, aseq.ungapped_to_gapped(start),
                        aseq.ungapped_to_gapped(end)])
        if blocks and self._first_sel_region_show:
            self._first_sel_region_show = False
            sv.status("ChimeraX selection region displayed.",
                follow_with="Settings..Regions controls this display.")
        sel_region.add_blocks(blocks)
        self.raise_region(sel_region)

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

    def show_ss(self, show):
        """show actual secondary structure"""
        from chimerax.atomic import Sequence
        helix_reg = self.get_region(self.ACTUAL_HELICES_REG_NAME, create=show,
                fill=Sequence.default_helix_fill_color,
                outline=Sequence.default_helix_outline_color)
        strand_reg = self.get_region(self.ACTUAL_STRANDS_REG_NAME, create=show,
                fill=Sequence.default_strand_fill_color,
                outline=Sequence.default_strand_outline_color)
        if helix_reg:
            helix_reg.shown = show
        if strand_reg:
            strand_reg.shown = show
        if not show:
            return
        helix_reg.clear(make_cb=False)  # callback will happen in
        strand_reg.clear(make_cb=False) # add_blocks below

        assoc_seqs = set()
        helices = []
        strands = []
        for aseq in self.seq_canvas.alignment.associations.values():
            assoc_seqs.add(aseq)
        for aseq in assoc_seqs:
            in_helix = in_strand = False
            for pos in range(len(aseq.ungapped())):
                is_helix = is_strand = False
                for match_map in aseq.match_maps.values():
                    try:
                        res = match_map[pos]
                    except KeyError:
                        continue
                    if res.is_helix:
                        is_helix = True
                    elif res.is_strand:
                        is_strand = True
                gapped = aseq.ungapped_to_gapped(pos)
                if is_helix:
                    if in_helix:
                        helices[-1][-1] = gapped
                    else:
                        helices.append([aseq, aseq, gapped, gapped])
                        in_helix = True
                else:
                    if in_helix:
                        in_helix = False
                if is_strand:
                    if in_strand:
                        strands[-1][-1] = gapped
                    else:
                        strands.append([aseq, aseq, gapped, gapped])
                        in_strand = True
                else:
                    if in_strand:
                        in_strand = False
        helix_reg.add_blocks(helices)
        strand_reg.add_blocks(strands)

    def unmap(self, *args):
        if self._mod_assoc_handler_id:
            self.seq_canvas.sv.triggers.deleteHandler(MOD_ASSOC,
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
        if self._drag_region and not self._drag_region.blocks:
            self.delete_region(self._drag_region)
            self._drag_region = None

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

    """
    def _delAssocCB(self, trigName, myData, delMatchMaps):
        for matchMap in delMatchMaps:
            key = (matchMap['mseq'], matchMap['aseq'])
            if key in self.associated_regions:
                self.deleteRegion(self.associated_regions[key][:])
    """

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

    def _key_press_cb(self, event):
        from Qt.QtCore import Qt
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            self.delete_region(self.selected())
            scene = self.seq_canvas.main_scene
            scene.update(scene.sceneRect())

    """
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
                        hr.name.startswith(self.seq_canvas.sv.GAP_REG_NAME_START)
                        or hr == self.get_region("ChimeraX selection")):
                            hr.shown = False
    """

    def _mouse_down_cb(self, event):
        from Qt.QtCore import Qt
        if event.button() == Qt.RightButton:
            """TODO
            if event.modifiers() & Qt.ShiftModifier:
                self._focus_cb(event, pref="region")
            else:
                self._focus_cb(event, pref="residue")
            """
            return
        pos = event.scenePos()
        self._start_x, self._start_y = pos.x(), pos.y()
        canvas = self.seq_canvas.main_view
        """TODO
        self._canvasHeight = canvas.winfo_height()
        self._canvasWidth = canvas.winfo_width()
        """
        self._clear_drag()

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

        from Qt.QtCore import Qt
        control_down = bool(event.modifiers() & Qt.ControlModifier)
        pos = event.scenePos()
        canvas_x, canvas_y = pos.x(), pos.y()
        if abs(canvas_x - self._start_x) > 1 or abs(canvas_y - self._start_y) > 1:
            block = self.seq_canvas.bounded_by(canvas_x, canvas_y, self._start_x, self._start_y,
                exclude_headers=True)
            if block[0] is None:
                self._clear_drag()
                return
            if not self._drag_region:
                rebuild_table = True
                """TODO
                rebuild_table = self.seqRegionMenu.index(Pmw.SELECT) == 0
                """
                settings = self.seq_canvas.sv.settings
                self._drag_region = self.new_region(blocks=[block], select=True,
                    outline=settings.new_region_border, fill=settings.new_region_interior,
                    cover_gaps=True, rebuild_table=rebuild_table)
                if not control_down and self._prev_drag:
                    # delete_region keeps sequence_regions and associated_regions
                    # up to date, direct destroy does not
                    self.delete_region(self._prev_drag, rebuild_table=rebuild_table)
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
                    from Qt.QtGui import QPen
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
        scene = self.seq_canvas.main_scene
        scene.update(scene.sceneRect())

    def _mouse_up_cb(self, event, double=False):
        canvas = self.seq_canvas.main_view
        if not self._drag_region:
            # maybe a region pick
            region = self._region(event)
            if region:
                #if double:
                #    self.enter()
                #else:
                if True:
                    self._toggle_active(region)
            else:
                # maybe a column pick
                col = self._column_pick(event)
                if col is not None:
                    residues = self._residues_in_block((self.seq_canvas.alignment.seqs[0],
                        self.seq_canvas.alignment.seqs[-1], col, col))
                    self.seq_canvas.sv.session.ui.main_window.select_by_mode(
                        " ".join([r.string(style="command") for r in sorted(residues)]))
        else:
            self._select_on_structures()
            self._prev_drag = self._drag_region
            rmsd = self._drag_region.rmsd
            if rmsd == None:
                from chimerax.mouse_modes import mod_key_info
                shift_name = mod_key_info("shift")[1]
                control_name = mod_key_info("control")[1]
                self.seq_canvas.sv.status(
                    "%s-drag to add to region; "
                    "%s-drag to start new region" % (shift_name.capitalize(), control_name))
                    #TODO:
                    #follow_with="Info->Region Browser to change region colors; "
                    #"%s left/right arrow to realign region" % control_name, follow_time=15)
            else:
                sv = self.seq_canvas.sv
                sv.status("Region RMSD: %.3f" % rmsd)
                sv.session.logger.info("%s region %s RMSD: %.3f\n"
                    % (sv.display_name ,self._drag_region, rmsd))
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
            self._toggle_active(region, destroyed=True)
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

    def _regionResiduesCB(self, event):
        region = self._region(event)
        if not region:
            return None
        residues = []
        for block in region.blocks:
            residues.extend(self._residues_in_block(block))
        return residues

    def _region_size_changed_cb(self, region):
        self.seq_canvas.sv._regions_tool_notification("rmsd", region)
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
        session = self.seq_canvas.sv.session
        sel_residues = self.region_residues(region)
        from chimerax.core.commands import run
        if sel_residues:
            from chimerax.atomic import concise_residue_spec
            run(session, "sel " + concise_residue_spec(session, sel_residues))
        else:
            run(session, "sel clear")
        self._sel_change_from_self = False

    def _sel_change_cb(self, _, changes):
        settings = self.seq_canvas.sv.settings
        sel_region = self.get_region("ChimeraX selection", create=True, read_only=True,
            fill=settings.sel_region_interior, outline=settings.sel_region_border)
        if self._sel_change_from_self:
            sel_region.clear()
        else:
            self.show_chimerax_selection()

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

    def _show_sel_cb(self):
        # also called from settings dialog
        from chimerax import atomic
        if self.seq_canvas.sv.settings.show_sel:
            self.show_chimerax_selection()
            from chimerax.core.selection import SELECTION_CHANGED
            self._sel_change_handler = self.seq_canvas.sv.session.triggers.add_handler(
                SELECTION_CHANGED, self._sel_change_cb)
        else:
            self._sel_change_handler.remove()
            self._sel_change_handler = None
            sel_region = self.get_region("ChimeraX selection")
            if sel_region:
                sel_region._destroy()

    def _toggle_active(self, region, select_on_structures=True, destroyed=False):
        if self._cur_region is not None and self._cur_region == region:
            if not destroyed:
                region.dehighlight()
            self._cur_region = None
        else:
            self._cur_region = region
            if not destroyed:
                self.highlight(region, select_on_structures=select_on_structures)

class RegionsTool:
    ENTIRE_ALIGNMENT_REGIONS = "entire alignment"

    def __init__(self, sv, tool_window):
        self.sv = sv
        self.tool_window = tool_window

        from Qt.QtCore import Qt
        from Qt.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMenu, QGroupBox
        from Qt.QtWidgets import QGridLayout, QCheckBox
        ui_area = tool_window.ui_area
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(5,5,5,5)
        ui_area.setLayout(layout)

        menu_layout = QHBoxLayout()
        layout.addLayout(menu_layout)
        menu_layout.addWidget(QLabel("Regions applicable to: "), alignment=Qt.AlignRight)
        self.seq_region_menubutton = mb = QPushButton(self.ENTIRE_ALIGNMENT_REGIONS)
        menu_layout.addWidget(mb, alignment=Qt.AlignLeft)
        menu = QMenu(mb)
        mb.setMenu(menu)
        menu.triggered.connect(self._seq_menu_cb)
        menu.aboutToShow.connect(self._fill_seq_region_menu)
        self.seq = None

        from chimerax.ui.widgets import ItemTable
        self.region_table = table = ItemTable(allow_user_sorting=False, session=tool_window.session,
            color_column_width=16)
        last = self.sv.settings.regions_tool_last_use
        from time import time
        now = self.sv.settings.regions_tool_last_use = time()
        short_titles = last != None and now - last < 777700 # about 3 months
        def blocks_to_val(region, index, func):
            blocks = region.blocks
            if not blocks:
                return None
            return func([b[index] for b in blocks])+1
        self.columns = {
            "active": table.add_column("A" if short_titles else "Active", "active",
                format=table.COL_FORMAT_BOOLEAN),
            "shown": table.add_column("S" if short_titles else "Shown", "shown",
                format=table.COL_FORMAT_BOOLEAN),
            "edge": table.add_column("B" if short_titles else "Border", self._get_edge,
                data_set=self._set_edge, format=table.COL_FORMAT_BOOLEAN),
            "edge color": table.add_column("\N{BALLOT BOX}", self._get_edge_color,
                data_set=self._set_edge_color, format=table.COL_FORMAT_OPAQUE_COLOR, color="forest green"),
            "fill": table.add_column("I" if short_titles else "Interior", self._get_fill,
                data_set=self._set_fill, format=table.COL_FORMAT_BOOLEAN),
            "fill color": table.add_column("\N{BLACK MEDIUM SQUARE}", self._get_fill_color,
                data_set=self._set_fill_color, format=table.COL_FORMAT_OPAQUE_COLOR, color="purple"),
            "name": table.add_column("Name", "display_name", editable=True, show_tooltips=True),
            "rmsd": table.add_column("RMSD", "rmsd", format="%.3f"),
            "start": table.add_column("Start", lambda r: blocks_to_val(r, 2, min), format="%d"),
            "end": table.add_column("End", lambda r: blocks_to_val(r, 3, max), format="%d"),
        }

        self.source_info = {}
        self.source_box = QGroupBox("Data source")
        source_layout = QVBoxLayout()
        source_layout.setContentsMargins(0,0,0,0)
        self.source_box.setLayout(source_layout)
        info = QLabel("List these kinds of regions...")
        from chimerax.ui import shrink_font
        shrink_font(info)
        source_layout.addWidget(info)
        self.sources_layout = QHBoxLayout()
        source_layout.addLayout(self.sources_layout)

        self._set_table_data(resize_columns=False)
        table.launch()
        table.selection_changed.connect(self._selection_changed)
        layout.addWidget(table, stretch=1)
        buttons_layout = QHBoxLayout()
        layout.addLayout(buttons_layout)
        for button_name, tool_tip in [
                ('Raise', "Raise region to top of drawing order"),
                ('Lower', "Lower region to bottom of drawing order"),
                ('Duplicate', None),
                ('Rename', None),
                ('Delete', None),
                ('Info', "Log info about region"),
                ]:
            button = QPushButton(button_name)
            if tool_tip:
                button.setToolTip(tool_tip)
            button.clicked.connect(lambda *args, f=self._button_cb, name=button_name: f(name))
            buttons_layout.addWidget(button, alignment=Qt.AlignCenter)

        layout.addWidget(self.source_box, alignment=Qt.AlignCenter)

        activities_layout = QHBoxLayout()
        layout.addLayout(activities_layout)

        actions_group = QGroupBox("Choosing in table")
        actions_layout = QHBoxLayout()
        actions_layout.setContentsMargins(0,0,0,0)
        actions_group.setLayout(actions_layout)
        self.activates_button = QCheckBox("activates  ")
        self.activates_button.setChecked(True)
        actions_layout.addWidget(self.activates_button)
        self.shows_button = QCheckBox("shows  ")
        self.shows_button.setChecked(True)
        actions_layout.addWidget(self.shows_button)
        self.hides_button = QCheckBox("hides others")
        self.hides_button.setChecked(True)
        actions_layout.addWidget(self.hides_button)
        disclaimer = QLabel(" (except missing structure,\n ChimeraX selection)")
        shrink_font(disclaimer)
        actions_layout.addWidget(disclaimer)
        activities_layout.addWidget(actions_group)

        activities_layout.addStretch(1)

        from chimerax.ui.widgets import TwoThreeStateCheckBox
        self.gaps_checkbox = TwoThreeStateCheckBox("Include gaps")
        self.gaps_checkbox.clicked.connect(self._gaps_cb)
        self.gaps_checkbox.setEnabled(False)
        activities_layout.addWidget(self.gaps_checkbox)

    def alignment_rmsd_update(self):
        self.region_table.update_column(self.columns["rmsd"], data=True)
        self.region_table.resizeColumnToContents(self.region_table.columns.index(self.columns["rmsd"]))

    def region_notification(self, category, region):
        if category in ("new", "delete"):
            self._set_table_data()
        elif category in ("raise", "lower"):
            table_regions = self.region_table.data
            all_regions = self.sv.region_manager.regions
            self.region_table.data = [r for r in all_regions if r in table_regions]
        elif category == "select":
            self.region_table.selected = [r for r in self.region_table.data if r == region]
        elif region in self.region_table.data:
            col = self.columns[category]
            self.region_table.update_cell(col, region)
            self.region_table.resizeColumnToContents(self.region_table.columns.index(col))

    @property
    def shown(self):
        return self.tool_window.shown

    @shown.setter
    def shown(self, show):
        self.tool_window.shown = show

    def _atomic_changes_cb(self, changes):
        if 'scene_coord changed' not in changes.structure_reasons():
            return
        for chain in self.sv.alignment.associations:
            if chain.structure in changes.modified_structures():
                self.region_table.update_column(self.columns["rmsd"], data=True)
                break

    def _button_cb(self, button_name):
        sel = self.region_table.selected
        if not sel:
            self.sv.status("No region chosen in table", color="red")
            return
        region = sel[0]
        mgr = self.sv.region_manager
        if button_name in ("Raise", "Lower"):
            getattr(mgr, button_name.lower() + "_region")(region)
        elif button_name == "Duplicate":
            mgr.copy_region(region)
        elif button_name == "Rename":
            self.region_table.edit_cell("Name", region)
        elif button_name == "Delete":
            mgr.delete_region(region)
        elif button_name == "Info":
            mgr.region_info(region)

    def _fill_seq_region_menu(self):
        menu = self.seq_region_menubutton.menu()
        menu.clear()
        from Qt.QtWidgets import QAction
        for i, label in enumerate([self.ENTIRE_ALIGNMENT_REGIONS]
                + [seq.name for seq in self.sv.alignment.seqs]):
            action = QAction(label, menu)
            action.setData(i)
            menu.addAction(action)

    def _gaps_cb(self, *args):
        for region in self.region_table.selected:
            region.set_cover_gaps(self.gaps_checkbox.isChecked())
        self.gaps_checkbox.setChecked(bool(self.gaps_checkbox.isChecked()))

    def _get_edge(self, region):
        return region.border_rgba is not None

    def _get_edge_color(self, region):
        if region.border_rgba is not None:
            return region.border_rgba
        if region.prev_border_rgba is not None:
            return region.prev_border_rgba
        return (0.5, 0.5, 0.5, 1.0)

    def _get_fill(self, region):
        return region.interior_rgba is not None

    def _get_fill_color(self, region):
        if region.interior_rgba is not None:
            return region.interior_rgba
        if region.prev_interior_rgba is not None:
            return region.prev_interior_rgba
        return (0.5, 0.5, 0.5, 1.0)

    def _selection_changed(self, *args):
        covers = set()
        sel = self.region_table.selected
        for region in sel:
            covers.add(region.cover_gaps)
            if self.shows_button.isChecked():
                region.shown = True
            if self.activates_button.isChecked():
                region.active = True
            if self.hides_button.isChecked():
                for r in self.sv.region_manager.regions:
                    if r not in sel and r.name and not (
                    r.name.startswith(self.sv.GAP_REGION_STRING) or r.name == "ChimeraX selection"):
                        r.shown = False
        if covers:
            self.gaps_checkbox.setEnabled(True)
            if len(covers) == 1:
                self.gaps_checkbox.setChecked(covers.pop())
            else:
                from Qt.QtCore import Qt
                self.gaps_checkbox.setCheckState(Qt.CheckState.PartiallyChecked)
        else:
            self.gaps_checkbox.setEnabled(False)

    def _seq_menu_cb(self, action):
        self.seq_region_menubutton.setText(action.text())
        self._set_table_data(menu_action=action)

    def _set_edge(self, region, edge):
        if edge:
            region.border_rgba = self._get_edge_color(region)
        else:
            region.border_rgba = None
            self.region_table.update_cell(self.columns["edge color"], region)

    def _set_edge_color(self, region, color):
        region.border_rgba = [ c/255.0 for c in color]
        self.region_table.update_cell(self.columns["edge"], region)

    def _set_fill(self, region, fill):
        if fill:
            region.interior_rgba = self._get_fill_color(region)
        else:
            region.interior_rgba = None
            self.region_table.update_cell(self.columns["fill color"], region)

    def _set_fill_color(self, region, color):
        region.interior_rgba = [ c/255.0 for c in color]
        self.region_table.update_cell(self.columns["fill"], region)

    def _set_table_data(self, *, menu_action=None, resize_columns=True):
        if menu_action:
            index = menu_action.data()
            if index == 0:
                self.seq = None
            else:
                self.seq = self.sv.alignment.seqs[index-1]
        regions = [r for r in self.sv.region_manager.regions if r.sequence == self.seq]

        # filter based on source (builtin, UniProt, etc.)
        all_sources = set([r.source for r in self.sv.region_manager.regions])
        sources = set([r.source for r in regions])
        for prev_source in list(self.source_info.keys()):
            if prev_source not in sources:
                if prev_source not in all_sources:
                    self.sources_layout.removeWidget(self.source_info[prev_source])
                    del self.source_info[prev_source]
                else:
                    self.source_info[prev_source].setHidden(True)
        from Qt.QtWidgets import QCheckBox
        for source in sources:
            if source in self.source_info:
                continue
            text = "built-in" if source is None else source
            cb = self.source_info[source] = QCheckBox(text)
            cb.setChecked(True)
            cb.clicked.connect(lambda *args, f=self._set_table_data: f())
            self.sources_layout.addWidget(cb)
        # if multiple sources, or if the single source is set to be hidden,
        # show the controls (and filter the regions)
        if sources and (len(sources) > 1 or not self.source_info[list(sources)[0]].isChecked()):
            for source, cb in self.source_info.items():
                cb.setHidden(source not in sources)
            self.source_box.setHidden(False)
            regions = [r for r in regions if r.source in sources and self.source_info[r.source].isChecked()]
        else:
            self.source_box.setHidden(True)

        self.region_table.update_column(self.columns["rmsd"], display=(self.seq is None))
        self.region_table.update_column(self.columns["start"], display=(self.seq is not None))
        self.region_table.update_column(self.columns["end"], display=(self.seq is not None))
        self.region_table.data = regions
        if resize_columns:
            self.region_table.resizeColumnsToContents()
        self.region_table.resizeRowsToContents()

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
"""

def get_rgba(color_info):
    if isinstance(color_info, str):
        from chimerax.core.colors import BuiltinColors
        return BuiltinColors[color_info].rgba
    return color_info

def rgba_to_qcolor(rgba):
    from Qt.QtGui import QBrush, QPen, QColor
    return QColor(*[int(255*chan + 0.5) for chan in rgba])
