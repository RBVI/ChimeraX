# vim: set expandtab ts=4 sw=4:

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

class GridCanvas:
    """'public' methods are only public to the ProfileGridsTool class.
       Access to GridCanvas functions is made through methods of the
       ProfileGridsTool class.
    """

    TEXT_MARGIN = 2

    def __init__(self, parent, pg, alignment, grid_data, weights):
        from Qt.QtWidgets import QGraphicsView, QGraphicsScene, QGridLayout, QShortcut, QHBoxLayout, QLabel
        from Qt.QtWidgets import QRadioButton
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
        self.font_descent = self.font_metrics.descent()
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
        self.main_scene.helpEvent = self.mouse_hover
        self.main_scene.mouseMoveEvent = self.mouse_move
        from Qt.QtWidgets import QToolTip
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
        # To show column number in status area as mouse is moved...
        self.main_view.setMouseTracking(True)
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
        layout.setRowStretch(2, 0)
        mouse_control_layout = QHBoxLayout()
        mouse_control_layout.setContentsMargins(0,0,0,0)
        mouse_control_layout.addWidget(QLabel("Mouse click:  "), alignment=Qt.AlignRight)
        self.mouse_selects = QRadioButton("selects residues / ")
        mouse_control_layout.addWidget(self.mouse_selects)
        self._choose_cell_text = "chooses cell"
        self.mouse_chooses = QRadioButton(self._choose_cell_text)
        mouse_control_layout.addWidget(self.mouse_chooses)
        mouse_control_layout.addWidget(QLabel(" (shift-click toggles)"), alignment=Qt.AlignLeft)
        self.mouse_selects.setChecked(True)
        layout.addLayout(mouse_control_layout, 2, 0, 1, 2, alignment=Qt.AlignHCenter|Qt.AlignTop)
        parent.setLayout(layout)
        #self.header_view.show()
        self.main_label_view.show()
        self.main_view.show()
        self.layout_alignment()
        self.chosen_cells = {}
        self.selection_items = {}
        self.update_selection()
        from chimerax.core.selection import SELECTION_CHANGED
        self.handlers = [ self.pg.session.triggers.add_handler(SELECTION_CHANGED, self.update_selection) ]

    def alignment_from_cells(self, viewer):
        seqs = self._check_cells()
        if len(seqs) == 1:
            seq_viewers = self.pg.session.alignments.registered_viewers("sequence")
            if viewer not in seq_viewers:
                self.pg.session.logger.warning(
                    "Cells only select a single sequence, showing in sequence viewer instead")
                viewer = True
        self.pg.session.alignments.new_alignment(seqs, None, name="grid subalignment", viewer=viewer)

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
            if note_name == self.alignment.NOTE_HDR_VALUES:
                if bounds is None:
                    bounds = (0, len(hdr)-1)
                self.refresh(hdr, *bounds)
            elif note_name == self.alignment.NOTE_HDR_NAME:
                label = self.header_label_items[hdr]
                start_label_rect = label.sceneBoundingRect()
                label.setText(hdr.name)
                end_label_rect = label.sceneBoundingRect()
                label.moveBy(start_label_rect.width() - end_label_rect.width(), 0)
                self._update_scene_rects()

    def destroy(self):
        for handler in self.handlers:
            handler.remove()

    def hide_header(self, header):
        self._clear_header_contents(header)
        label_item = self.header_label_items[header]
        del self.header_label_items[header]
        label_item.hide()
        self.header_label_scene.removeItem(label_item)
        after_removed = False
        width, height = self.font_pixels
        for disp_hdr in self.displayed_headers:
            if disp_hdr == header:
                after_removed = True
            elif after_removed:
                self.header_groups[disp_hdr].moveBy(0, -height)
                self.header_label_items[disp_hdr].moveBy(0, -height)
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
        self._cell_text_infos = []
        for i in range(rows):
            if i in self.empty_rows:
                continue
            for j in range(columns):
                x = j * width
                val = self.grid_data[i,j]
                fraction = val / divisor
                # The "cell chosen" contrast color has to change if this color changes
                non_blue = int(255 * (1.0 - fraction) + 0.5)
                fill_color = QColor(non_blue, non_blue, 255)
                self.main_scene.addRect(x, y, width, height, brush=QBrush(fill_color))
                if val > 0.0:
                    text_rgb = contrast_with((non_blue/255.0, non_blue/255.0, 1.0))
                    text_val = self._cell_text(val, fraction)
                    cell_text = self.main_scene.addSimpleText(text_val, self.font)
                    self._center_cell_text(cell_text, x, y, y_adjust)
                    cell_text.setBrush(QBrush(QColor(*[int(255 * channel + 0.5) for channel in text_rgb])))
                    self._cell_text_infos.append((cell_text, x, y, y_adjust, val, fraction))
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

    def list_from_cells(self):
        seqs = self._check_cells()
        _SeqList(self.pg.session, seqs).show()

    def mouse_click(self, event):
        from Qt.QtCore import Qt
        shifted = event.modifiers() & Qt.ShiftModifier

        residues, row, col = self._residues_for_event(event)
        if self.mouse_selects.isChecked():
            if not residues:
                return
            final_cmd = None
            if shifted:
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
        else:
            if shifted:
                try:
                    item = self.chosen_cells[(row, col)]
                except KeyError:
                    pass # fall through to choosing the cell, below
                else:
                    item.hide()
                    self.main_scene.removeItem(item)
                    del self.chosen_cells[(row, col)]
                    return
            else:
                for item in self.chosen_cells.values():
                    item.hide()
                    self.main_scene.removeItem(item)
                self.chosen_cells.clear()
            self._choose_cell(row, col)

    def mouse_hover(self, event):
        if event.type() != event.GraphicsSceneHelp:
            return
        from Qt.QtWidgets import QToolTip
        residues, row, col = self._residues_for_event(event)
        if not residues:
            QToolTip.hideText()
            return
        from chimerax.atomic import concise_residue_spec
        self.main_view.setToolTip(concise_residue_spec(self.pg.session, residues))
        QToolTip.showText(event.screenPos(), self.main_view.toolTip())

    def mouse_move(self, event):
        residues, row, col = self._residues_for_event(event)
        if col is not None:
            self.pg.status("Column %d" % (col+1), secondary=True)

    def refresh(self, seq, left=0, right=None):
        if seq not in self.alignment.headers:
            # Since grids typically don't contain StructureSeqs, this won't
            # happen often, so do the minimum
            from chimerax.core.errors import UserError
            raise UserError("Profile Grid does not support updating sequence contents.")
        if not seq.shown:
            return
        if right is None:
            right = len(self.alignment.seqs[0])-1
        self._clear_header_contents(seq)
        self._fill_header_contents(seq)
        self._update_scene_rects()

    def restore_state(self, state):
        for row, col in state['chosen cells']:
            self._choose_cell(row, col)
        check_box = self.mouse_selects if state['mouse selects'] else self.mouse_chooses
        check_box.setChecked(True)

    def show_header(self, header):
        self.displayed_headers.append(header)
        group = self._fill_header_contents(header)
        bbox = group.sceneBoundingRect()
        self.header_label_items[header] = label = self.header_label_scene.addSimpleText(header.name,
            font=self.font)
        label_rect = label.sceneBoundingRect()
        group_rect = group.boundingRect()
        label.setPos(-label_rect.width(), group_rect.y() - label_rect.height())
        self.header_view.show()
        self._update_scene_rects()

    def state(self):
        return {
            'chosen cells': list(self.chosen_cells.keys()),
            'mouse selects': self.mouse_selects.isChecked(),
        }

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
            match_map = self.alignments.match_maps[aseq][chain]
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

    def _cell_text(self, val, fraction):
        cell_text_type = self.pg.settings.cell_text
        if cell_text_type == "percentage":
            digits = self.pg.settings.percent_decimal_places
            text_val = str(round(100  * fraction, digits if digits else None))
        elif cell_text_type == "count":
            text_val = str(round(val))
        else:
            text_val = ""
        return text_val

    def _center_cell_text(self, cell_text, x, y, y_adjust):
        width, height = self.font_pixels
        cell_text.setPos(x, y)
        cell_text.setZValue(1)
        bbox = cell_text.boundingRect()
        cell_text.moveBy((width - bbox.width())/2, y_adjust + (height - bbox.height())/2)

    def _check_cells(self):
        from chimerax.core.errors import UserError
        if not self.chosen_cells:
            raise UserError("No grid cells are chosen.\n"
                "Choose cells by changing mouse-click mode at bottom of window to '%s'\n"
                " and then clicking on desired cell(s)" % self._choose_cell_text)

        # since cells in the same column 'union' together, but columns intersect, organize by column...
        by_col = {}
        for row, col in self.chosen_cells.keys():
            by_col.setdefault(col, []).append(row)
        seqs = set(self.alignment.seqs)
        for col, rows in by_col.items():
            col_seqs = set()
            for row in rows:
                col_seqs.update(self._sequences_at(row, col))
            seqs &= col_seqs
        # in same order though
        aln_seqs = [seq for seq in self.alignment.seqs if seq in seqs]
        if not aln_seqs:
            raise UserError("No sequences match the chosen cells")
        return aln_seqs

    def _choose_cell(self, row, col):
        from Qt.QtGui import QPen, QColor, QPolygonF
        from Qt.QtCore import QPointF
        pen = QPen(QColor(255, 147, 0))
        pen.setWidth(3)
        width, height = self.font_pixels
        left_x = col * width
        mid_x = left_x + width/2
        right_x = left_x + width
        top_y = row * height
        mid_y = top_y + height/2
        bottom_y = top_y + height
        self.chosen_cells[(row, col)] = self.main_scene.addPolygon(QPolygonF([QPointF(x, y) for x,y in
            [(left_x, mid_y), (mid_x, top_y), (right_x, mid_y), (mid_x, bottom_y), (left_x, mid_y)]]), pen)

    def _clear_header_contents(self, header):
        header_group = self.header_groups[header]
        del self.header_groups[header]
        for item in header_group.childItems():
            item.hide()
            self.header_scene.removeItem(item)
        self.header_scene.destroyItemGroup(header_group)

    def _fill_header_contents(self, header):
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
                text.setPos(x - rect.width()/2, y - (height - self.TEXT_MARGIN + rect.height())/2)
                text.setBrush(QBrush(color))
                items.append(text)
            elif val != None and val > 0.0:
                display_height = height - self.TEXT_MARGIN
                items.append(self.header_scene.addRect(x - width/2, y - self.TEXT_MARGIN/2,
                    width, -val * display_height, brush=QBrush(color)))
            x += width

        self.header_groups[header] = group = self.header_scene.createItemGroup(items);
        return group

    def _residues_at(self, grid_row, grid_col):
        residues = []
        for seq in self._sequences_at(grid_row, grid_col):
            for match_map in self.alignment.match_maps[seq].values():
                try:
                    residues.append(match_map[seq.gapped_to_ungapped(grid_col)])
                except KeyError:
                    continue
        return residues

    def _sequences_at(self, grid_row, grid_col):
        seqs = []
        row_label = self.existing_row_labels[grid_row]
        for seq in self.alignment.seqs:
            if seq.characters[grid_col].upper() == row_label:
                seqs.append(seq)
        return seqs

    def _residues_for_event(self, event):
        width, height = self.font_pixels
        raw_rows, grid_columns = self.grid_data.shape
        grid_rows = raw_rows - len(self.empty_rows)
        pos = event.scenePos()
        row = int(pos.y() / height)
        if row < 0 or row > grid_rows - 1:
            return None, None, None
        col = int(pos.x() / width)
        if col < 0 or col > grid_columns - 1:
            return None, None, None
        return self._residues_at(row, col), row, col

    def _update_cell_texts(self):
        for cell_text, *pos_args, val, fraction in self._cell_text_infos:
            cell_text.setText(self._cell_text(val, fraction))
            self._center_cell_text(cell_text, *pos_args)

    def _update_scene_rects(self):
        # have to play with setViewportMargins to get correct scrolling...
        #self.main_view.setViewportMargins(0, 0, 0, -20)
        mbr = self.main_scene.itemsBoundingRect()
        self.main_scene.setSceneRect(self.main_scene.itemsBoundingRect())
        # For scrolling to work right, ensure that vertical size of main_label_scene is the same as main_scene
        # and that the horizontal size of the header_scene is the same as the main_scene
        lbr = self.main_label_scene.itemsBoundingRect()
        hlbr = self.header_label_scene.itemsBoundingRect()
        label_width = max(lbr.width(), hlbr.width())
        mbr = self.main_scene.itemsBoundingRect()
        y = min(lbr.y(), mbr.y())
        height = max(lbr.y() + lbr.height() - y, mbr.y() + mbr.height() - y)
        mr = self.main_scene.sceneRect()
        hbr = self.header_scene.itemsBoundingRect()
        self.main_label_scene.setSceneRect(lbr.x() + lbr.width() - label_width, y, label_width, height)
        self.main_scene.setSceneRect(mbr.x(), y, mbr.width(), height)
        self.header_scene.setSceneRect(mr.x(), hbr.y(),
            mr.width() + self.main_view.verticalScrollBar().size().width(), hbr.height())
        self.header_label_scene.setSceneRect(hlbr.x() + hlbr.width() - label_width,
            hlbr.y() + (hbr.height() - hlbr.height())/2, label_width, hbr.height())
        from math import ceil
        max_header_height = ceil(max(hbr.height(), hlbr.height())) + 7
        self.header_view.setMaximumHeight(max_header_height)
        self.header_label_view.setMaximumHeight(max_header_height)

        # Apparently the height of the horizontal scrollbar gets added to main view at some point,
        # need to compensate
        from Qt.QtCore import QTimer, Qt
        def adjust_scrollbars(mlv=self.main_label_view, mv=self.main_view):
            sb1 = mlv.verticalScrollBar()
            sb2 = mv.verticalScrollBar()
            min_val = min(sb1.minimum(), sb2.minimum())
            max_val = max(sb1.maximum(), sb2.maximum())
            sb1.setRange(min_val, max_val)
            sb2.setRange(min_val, max_val)
            # on Mac, if the user has their scrollbar policy as "always on", there might be a horizontal
            # scrollbar on the main canvas and not on the label canvas, which makes the viewports heights
            # different, so they don't scroll in sync; compensate by adding horizontal scroller to labels
            lvr = mlv.viewport().rect()
            mvr = mv.viewport().rect()
            if lvr.height() > mvr.height():
                mlv.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
                def right_scroll(mlv=mlv):
                    hsb = mlv.horizontalScrollBar()
                    hsb.setValue(hsb.maximum())
                QTimer.singleShot(100, right_scroll)
        QTimer.singleShot(100, adjust_scrollbars)

_seq_lists = [] # hold references so the lists aren't immediately destroyed
from Qt.QtWidgets import QDialog
class _SeqList(QDialog):
    help = None

    def __init__(self, session, seqs):
        super().__init__()
        _seq_lists.append(self)
        self.session = session
        self.setWindowTitle("Cell-Chosen Sequence List")
        self.setSizeGripEnabled(True)
        from Qt.QtWidgets import QVBoxLayout, QTextEdit, QHBoxLayout, QPushButton, QLabel, QWidget
        from Qt.QtCore import Qt
        layout = QVBoxLayout()
        list_widget = QTextEdit('<br>'.join([seq.name for seq in seqs]))
        list_widget.setReadOnly(True)
        layout.addWidget(list_widget, stretch=1)

        centering_widget = QWidget()
        button_layout = QVBoxLayout()
        button_layout.setSpacing(0)
        button_layout.setContentsMargins(0,0,0,0)
        centering_widget.setLayout(button_layout)
        layout.addWidget(centering_widget, alignment=Qt.AlignCenter)

        centering_widget = QWidget()
        log_layout = QHBoxLayout()
        log_layout.setSpacing(0)
        log_layout.setContentsMargins(0,0,0,0)
        centering_widget.setLayout(log_layout)
        log_button = QPushButton("Copy")
        log_button.clicked.connect(lambda *args, seqs=seqs, f=self._log_sequences: f(seqs))
        log_layout.addWidget(log_button, alignment=Qt.AlignRight)
        log_layout.addWidget(QLabel(" sequence names to log"), alignment=Qt.AlignLeft)
        button_layout.addWidget(centering_widget, alignment=Qt.AlignCenter)

        centering_widget = QWidget()
        file_layout = QHBoxLayout()
        file_layout.setSpacing(0)
        file_layout.setContentsMargins(0,0,0,0)
        centering_widget.setLayout(file_layout)
        file_button = QPushButton("Save")
        file_button.clicked.connect(lambda *args, seqs=seqs, f=self._save_sequences: f(seqs))
        file_layout.addWidget(file_button, alignment=Qt.AlignRight)
        file_layout.addWidget(QLabel(" sequence names to file"), alignment=Qt.AlignLeft)
        button_layout.addWidget(centering_widget, alignment=Qt.AlignCenter)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Close)
        bbox.rejected.connect(self.close)
        layout.addWidget(bbox)

        self.setLayout(layout)

    def closeEvent(self, event):
        _seq_lists.remove(self)
        return super().closeEvent(event)

    def _log_sequences(self, seqs):
        self.session.logger.info('<br>'.join(["<br><b>Chosen Profile Grid Sequences</b>"]
            + [seq.name for seq in seqs]) + '<br>', is_html=True)

    def _save_sequences(self, seqs):
        from Qt.QtWidgets import QFileDialog
        file_name, file_type = QFileDialog.getSaveFileName(caption="Choose Sequence-Name Save File")
        if file_name:
            with open(file_name, 'w') as f:
                for seq in seqs:
                    print(seq.name, file=f)

