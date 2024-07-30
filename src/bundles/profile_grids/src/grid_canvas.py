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

class GridCanvas:
    """'public' methods are only public to the ProfileGridsTool class.
       Access to GridCanvas functions is made through methods of the
       ProfileGridsTool class.
    """

    TEXT_MARGIN = 2

    def __init__(self, parent, pg, alignment, grid_data, weights):
        from Qt.QtWidgets import QGraphicsView, QGraphicsScene, QGridLayout, QShortcut
        from Qt.QtCore import Qt, QSize

        self.pg = pg
        self.alignment = alignment
        self.grid_data = grid_data
        self.weights = weights

        import string
        self.row_labels = list(string.ascii_uppercase) + ['?', 'gap', 'misc']
        import numpy
        self.empty_rows = numpy.where(~self.grid_data.any(axis=1))[0]
        self.existing_row_labels = [rl for i, rl in enumerate(self.row_labels) if i not in self.empty_rows]
        from Qt.QtGui import QFont, QFontMetrics, QPalette
        self.font = QFont("Helvetica")
        self.font_metrics = QFontMetrics(self.font)
        self.max_label_width = 0
        for i, text in enumerate(self.row_labels):
            if i in self.empty_rows:
                continue
            self.max_label_width = max(self.max_label_width, self.font_metrics.horizontalAdvance(text + ' '))
        self.max_main_label_width = self.max_label_width

        #palette = QPalette()
        #palette.setColor(QPalette.Window, Qt.white)
        #parent.setAutoFillBackground(True)
        #parent.setPalette(palette)

        self.main_label_scene = QGraphicsScene()
        """
        self.main_label_scene.setBackgroundBrush(Qt.lightGray)
        """
        self.main_label_scene.setBackgroundBrush(Qt.white)
        self.main_label_view = QGraphicsView(self.main_label_scene)
        self.main_label_view.setAlignment(Qt.AlignRight|Qt.AlignTop)
        self.main_label_view.setAttribute(Qt.WA_AlwaysShowToolTips)
        self.header_scene = QGraphicsScene()
        """
        self.header_scene.setBackgroundBrush(Qt.lightGray)
        """
        self.header_scene.setBackgroundBrush(Qt.white)
        self.header_view = QGraphicsView(self.header_scene)
        self.header_view.setAttribute(Qt.WA_AlwaysShowToolTips)
        self.header_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.header_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.header_view.setAlignment(Qt.AlignLeft|Qt.AlignBottom)
        self.header_label_scene = QGraphicsScene()
        """
        self.header_label_scene.setBackgroundBrush(Qt.lightGray)
        """
        self.header_label_scene.setBackgroundBrush(Qt.white)
        self.header_label_view = QGraphicsView(self.header_label_scene)
        self.header_label_view.setAlignment(Qt.AlignRight|Qt.AlignBottom)
        self.header_label_view.setAttribute(Qt.WA_AlwaysShowToolTips)

        self.main_scene = QGraphicsScene()
        self.main_scene.setBackgroundBrush(Qt.white)
        self.main_scene.mouseReleaseEvent = self.mouse_click
        """if gray background desired...
        ms_brush = self.main_scene.backgroundBrush()
        from Qt.QtGui import QColor
        ms_color = QColor(240, 240, 240) # lighter gray than "lightGray"
        ms_brush.setColor(ms_color)
        self.main_scene.setBackgroundBrush(ms_color)
        """
        self.main_view = QGraphicsView(self.main_scene)
        self.main_view.setAttribute(Qt.WA_AlwaysShowToolTips)
        #self.main_view.setViewportMargins(0, 0, 0, -20)
        #from Qt.QtWidgets import QFrame
        #self.main_view.setFrameStyle(QFrame.NoFrame)
        #self.main_view.setMouseTracking(True)
        main_vsb = self.main_view.verticalScrollBar()
        label_vsb = self.main_label_view.verticalScrollBar()
        main_vsb.valueChanged.connect(label_vsb.setValue)
        label_vsb.valueChanged.connect(main_vsb.setValue)
        main_hsb = self.main_view.horizontalScrollBar()
        header_hsb = self.header_view.horizontalScrollBar()
        main_hsb.valueChanged.connect(header_hsb.setValue)
        header_hsb.valueChanged.connect(main_hsb.setValue)
        #self.emphasis_font = QFont(self.font)
        #self.emphasis_font.setBold(True)
        #self.emphasis_font_metrics = QFontMetrics(self.emphasis_font)
        digits = self.pg.settings.percent_decimal_places
        wide_string = "100" if digits == 0 else "100." + '0' * digits
        font_width, font_height = self.font_metrics.horizontalAdvance(wide_string), self.font_metrics.height()
        self.main_label_view.setMinimumHeight(font_height)
        self.main_view.setMinimumHeight(font_height)
        # pad font a little...
        self.font_pixels = (font_width + self.TEXT_MARGIN, font_height + self.TEXT_MARGIN)
        layout = QGridLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        layout.addWidget(self.header_label_view, 0, 0, alignment=Qt.AlignRight | Qt.AlignBottom)
        layout.addWidget(self.header_view, 0, 1, alignment=Qt.AlignLeft | Qt.AlignBottom)
        layout.addWidget(self.main_label_view, 1, 0, alignment=Qt.AlignRight | Qt.AlignTop)
        layout.addWidget(self.main_view, 1, 1, alignment=Qt.AlignLeft | Qt.AlignTop)
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(0, 0)
        layout.setRowStretch(1, 1)
        parent.setLayout(layout)
        #self.header_view.show()
        self.main_label_view.show()
        self.main_view.show()
        self.layout_alignment()
        self.selection_items = {}
        self.update_selection()
        from chimerax.core.selection import SELECTION_CHANGED
        self.handlers = [ self.pg.session.triggers.add_handler(SELECTION_CHANGED, self.update_selection) ]

    def alignment_notification(self, note_name, note_data):
        alignment = self.alignment
        if note_name == alignment.NOTE_MOD_ASSOC:
            self.update_selection()
        '''
        if note_name == self.alignment.NOTE_REF_SEQ:
            self.lead_block.rerule()
        elif note_name == self.alignment.NOTE_SEQ_CONTENTS:
            self.refresh(note_data)
        elif note_name == self.alignment.NOTE_REALIGNMENT:
            # headers are notified before us, so they should be "ready to go"
            self.sv.region_browser.clear_regions()
            self._reformat()
        '''
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
            #TODO
            raise NotImplementedError("Updating header label/values not implemented")
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
                    self.main_label_scene.update()
                else:
                    self._reformat()

    def destroy(self):
        for handler in self.handlers:
            handler.remove()

    def hide_header(self, header):
        header_group = self.header_groups[header]
        del self.header_groups[header]
        for item in header_group.childItems():
            item.hide()
            self.header_scene.removeItem(item)
        self.header_scene.destroyItemGroup(header_group)
        label_item = self.header_label_items[header]
        del self.header_label_items[header]
        label_item.hide()
        self.header_label_scene.removeItem(label_item)
        self.displayed_headers.remove(header)
        self._update_scene_rects()

    def layout_alignment(self):
        #NOTE: maybe group each header line (QGraphicsItemGroup) to make them easier to move
        rows, columns = self.grid_data.shape
        if rows != len(self.row_labels):
            raise AssertionError("Expected %d rows, got %d" % (len(self.row_labels), rows))
        width, height = self.font_pixels
        divisor = sum(self.weights)
        from Qt.QtGui import QColor, QBrush
        from chimerax.core.colors import contrast_with
        y = 0
        # adjust for rectangle outline width / inter-line spacing
        y_adjust = 2
        for i in range(rows):
            if i in self.empty_rows:
                continue
            for j in range(columns):
                x = j * width
                val = self.grid_data[i,j]
                fraction = val / divisor
                non_blue = int(255 * (1.0 - fraction) + 0.5)
                fill_color = QColor(non_blue, non_blue, 255)
                self.main_scene.addRect(x, y, width, height, brush=QBrush(fill_color))
                if val > 0.0:
                    text_rgb = contrast_with((non_blue/255.0, non_blue/255.0, 1.0))
                    text_val = str(int(100  * fraction + 0.5))
                    cell_text = self.main_scene.addSimpleText(text_val, self.font)
                    cell_text.moveBy(x, y)
                    bbox = cell_text.boundingRect()
                    cell_text.moveBy((width - bbox.width())/2, y_adjust + (height - bbox.height())/2)
                    cell_text.setBrush(QBrush(QColor(*[int(255 * channel + 0.5) for channel in text_rgb])))
            label_text = self.main_label_scene.addSimpleText(self.row_labels[i], self.font)
            label_width = self.font_metrics.horizontalAdvance(self.row_labels[i] + ' ')
            label_text.moveBy((self.max_label_width - label_width) / 2, y + y_adjust)
            y += height
        self.header_groups = {}
        self.header_label_items = {}
        self.displayed_headers = []
        for hdr in self.alignment.headers:
            if hdr.shown:
                self.show_header(hdr)
        self._update_scene_rects()
        #TODO: everything else
        return
        raise NotImplementedError("layout_alignment")
        initial_headers = [hd for hd in self.alignment.headers if hd.shown]
        self.label_width = _find_label_width(self.alignment.seqs + initial_headers,
            self.sv.settings, self.font_metrics, self.emphasis_font_metrics, SeqBlock.label_pad)

        self._show_ruler = self.sv.settings.alignment_show_ruler_at_startup and len(self.alignment.seqs) > 1
        self.line_width = self.line_width_from_settings()
        self.numbering_widths = self.find_numbering_widths(self.line_width)
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

    def mouse_click(self, event):
        from Qt.QtCore import Qt
        width, height = self.font_pixels
        raw_rows, grid_columns = self.grid_data.shape
        grid_rows = raw_rows - len(self.empty_rows)
        pos = event.scenePos()
        row = int(pos.y() / height)
        if row < 0 or row > grid_rows - 1:
            return
        col = int(pos.x() / width)
        if col < 0 or col > grid_columns - 1:
            return
        residues = self._residues_at(row, col)
        final_cmd = None
        if event.modifiers() & Qt.ShiftModifier:
            if not residues:
                return
            if (row, col) in self.selection_items:
                cmd = "sel subtract"
            else:
                cmd = "sel add"
        else:
            if residues:
                cmd = "sel"
            else:
                final_cmd = "sel clear"
        if final_cmd is None:
            from chimerax.atomic import concise_residue_spec
            final_cmd = cmd + ' ' + concise_residue_spec(self.pg.session, residues)
        from chimerax.core.commands import run
        run(self.pg.session, final_cmd)

    def refresh(self, seq, left=0, right=None, update_attrs=True):
        raise NotImplementedError("refresh")
        if seq in self.alignment.headers and not seq.shown:
            return
        if right is None:
            right = len(self.alignment.seqs[0])-1
        self.lead_block.refresh(seq, left, right)
        self.main_scene.update()

    def show_header(self, header):
        self.displayed_headers.append(header)
        width, height = self.font_pixels
        x = width / 2
        #if not self.header_groups:
        #    y = 0
        #else:
        #    y = max([grp.boundingRect().y() for grp in self.header_groups.values()]) + height + 2
        y = len(self.header_groups) * height
        items = []
        if hasattr(header, 'depiction_val'):
            val_func = lambda i, hdr=header: hdr.depiction_val(i)
        else:
            val_func = lambda i, hdr=header: hdr[i]
        from chimerax.alignment_headers import position_color_to_qcolor as qcolor
        from Qt.QtGui import QBrush
        for i in range(len(header)):
            val = val_func(i)
            color = qcolor(header.position_color(i))
            if isinstance(val, str):
                text = self.header_scene.addSimpleText(val, font=self.font)
                rect = text.sceneBoundingRect()
                text.setPos(x - rect.width()/2, y+2)
                text.setBrush(QBrush(color))
                items.append(text)
            elif val != None and val > 0.0:
                items.append(self.header_scene.addRect(x - width/2, y+height, width, -val * height,
                    brush=QBrush(color)))
            x += width

        self.header_groups[header] = group = self.header_scene.createItemGroup(items);
        self.header_label_items[header] = label = self.header_label_scene.addSimpleText(header.name,
            font=self.font)
        label_rect = label.sceneBoundingRect()
        group_rect = group.boundingRect()
        label.setPos(-label_rect.width(), group_rect.y() - label_rect.height())
        self.header_view.show()
        self._update_scene_rects()

    def update_selection(self, *args):
        for item in self.selection_items.values():
            self.main_scene.removeItem(item)
        self.selection_items.clear()
        from chimerax.atomic import selected_chains, selected_residues
        sel_chains = set(selected_chains(self.pg.session))
        if not sel_chains:
            return
        sel_residues = set(selected_residues(self.pg.session))
        needs_highlight = set()
        chars_to_rows = { c:i for i,c in enumerate(self.existing_row_labels) }
        for chain, aseq in self.alignment.associations.items():
            if chain not in sel_chains:
                continue
            match_map = aseq.match_maps[chain]
            for r in chain.existing_residues:
                if r not in sel_residues:
                    continue
                try:
                    ungapped_seq_index = match_map[r]
                except KeyError:
                    continue
                gapped_seq_index = aseq.ungapped_to_gapped(ungapped_seq_index)
                char = aseq[gapped_seq_index].upper()
                if char.isupper() or char == '?':
                    row = chars_to_rows[char]
                else:
                    row = chars_to_rows['misc']
                needs_highlight.add((row, gapped_seq_index))
        from Qt.QtGui import QPen, QColor
        pen = QPen(QColor(87, 202, 35))
        pen.setWidth(3)
        width, height = self.font_pixels
        for row, col in needs_highlight:
            self.selection_items[(row, col)] = self.main_scene.addRect(
                col * width, row * height, width, height, pen=pen)

    def _residues_at(self, grid_row, grid_col):
        residues = []
        row_label = self.existing_row_labels[grid_row]
        for seq in self.alignment.seqs:
            if seq.characters[grid_col].upper() != row_label:
                continue
            for match_map in seq.match_maps.values():
                try:
                    residues.append(match_map[seq.gapped_to_ungapped(grid_col)])
                except KeyError:
                    continue
        return residues

    def _update_scene_rects(self):
        # have to play with setViewportMargins to get correct scrolling...
        #self.main_view.setViewportMargins(0, 0, 0, -20)
        mbr = self.main_scene.itemsBoundingRect()
        self.main_scene.setSceneRect(self.main_scene.itemsBoundingRect())
        # For scrolling to work right, ensure that vertical size of main_label_scene is the same as main_scene
        # and that the horizontal size of the header_scene is the same as the main_scene
        lbr = self.main_label_scene.itemsBoundingRect()
        mbr = self.main_scene.itemsBoundingRect()
        y = min(lbr.y(), mbr.y())
        height = max(lbr.y() + lbr.height() - y, mbr.y() + mbr.height() - y)
        mr = self.main_scene.sceneRect()
        hbr = self.header_scene.itemsBoundingRect()
        self.main_label_scene.setSceneRect(lbr.x(), y, lbr.width(), height)
        self.main_scene.setSceneRect(mbr.x(), y, mbr.width(), height)
        self.header_scene.setSceneRect(mr.x(), hbr.y(), mr.width(), hbr.height())
        from math import ceil
        max_header_height = ceil(max(hbr.height(), self.header_label_scene.itemsBoundingRect().height())) + 7
        self.header_view.setMaximumHeight(max_header_height)
        self.header_label_view.setMaximumHeight(max_header_height)
