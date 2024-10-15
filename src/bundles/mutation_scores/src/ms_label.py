# Try to replace simple text label with a custom image.
def mutation_scores_label(session, residues, score_name = None, scores_name = None, subtract_fit = None,
                          range = None, palette = None, no_data_color = (180,180,180,255),
                          height = 1.5, offset = (0,0,3), on_top = False):

    messages = []
    for chain, cresidues in _residues_by_chain(residues):
        from .ms_data import mutation_scores
        scores = mutation_scores(session, scores_name)
        score_values = scores.score_values(score_name, subtract_fit = subtract_fit)
        from chimerax.surface.colorvol import _use_full_range, _colormap_with_range
        vrange = score_values.value_range()
        r = vrange if _use_full_range(range, palette) else range
        colormap = _colormap_with_range(palette, r)
        count = 0
        for res in cresidues:
            mut_colors = {to_aa:colormap.interpolated_rgba8([value])[0]
                          for from_aa, to_aa, value in score_values.mutation_values(res.number)}
            if mut_colors:
                label_residue(res, mut_colors, no_data_color, height = height, offset = offset, on_top = on_top)
                count += 1

        message = f'Added {count} residue labels to chain {chain} for {score_name} ({"%.3g"%vrange[0]} - {"%.3g"%vrange[1]}), no mutation scores for {len(residues) - count} residues'
        session.logger.info(message)

def _residues_by_chain(residues):
    cres = {}
    for r in residues:
        c = r.chain
        if c is None:
            continue
        if c in cres:
            cres[c].append(r)
        else:
            cres[c] = [r]
    from chimerax.atomic import Residues
    return [(c, Residues(res)) for c,res in cres.items()]

def label_residue(residue, mutation_colors, no_data_color, height = 1.5, offset = (0,0,3), on_top = False):
    # Replace _label_image method of ObjectLabel to supply my own RGBA array
    from chimerax.label.label3d import labels_model, ResidueLabel
    view = residue.structure.session.main_view
    lm = labels_model(residue.structure, create = True)
    settings = {'height':height, 'offset':offset}
    lm.add_labels([residue], ResidueLabel, view, settings, on_top)
    ol = lm.labels([residue])[0]
    title = f'{residue.one_letter_code}{residue.number}'
    def label_image(self, title = title, mutation_colors = mutation_colors, no_data_color = no_data_color):
        rgba = label_rgba(title, mutation_colors, no_data_color)
        h,w = rgba.shape[:2]
        self._label_size = w,h
        return rgba
    from types import MethodType
    ol._label_image = MethodType(label_image, ol)
    lm.update_labels()

def label_rgba(title, mutation_colors, no_data_color):
    amino_acids = 'PRKHDEFWYNQCSTILVMGA'
    colors = [mutation_colors.get(r, no_data_color) for r in amino_acids]
    from Qt.QtGui import QImage, QPainter, QFont, QColor, QBrush, QPen
    wc,hc = 40,40	# Cell size in pixels
    font_size = 40
    xpad, ypad = 5, 5	# Font offset pixels
    rows, columns = 4, 5
    w,h = columns*wc, (rows+1)*hc
    text_color = (0,0,0,255)
    font = 'Helvetica'
    p = QPainter()
    ti = QImage(w, h, QImage.Format.Format_ARGB32)
    p.begin(ti)
    p.setCompositionMode(p.CompositionMode_Source)
    from Qt.QtCore import Qt
    pbr = QBrush(Qt.SolidPattern)
    p.setBrush(pbr)
    ppen = QPen(Qt.NoPen)
    p.setPen(ppen)
    # Title color
    pbr.setColor(QColor(*no_data_color))
    p.fillRect(0,0,w,hc,pbr)
    # Grid colors
    for r in range(rows):
        for c in range(columns):
            x,y = c*wc, (r+1)*hc
            rgba8 = tuple(colors[c + r*columns])
            pbr.setColor(QColor(*rgba8))
            p.fillRect(x,y,wc,hc,pbr)
    f = QFont(font)
    f.setPixelSize(font_size)
    p.setFont(f)
    c = QColor(*text_color)
    p.setPen(c)
    p.drawText(wc+xpad, hc-ypad, title)	# Title text
    # Grid letters
    for r in range(rows):
        for c in range(columns):
            x,y = c*wc + xpad, (r+1)*hc - ypad
            p.drawText(x, y+hc, amino_acids[c + r*columns])

    # Convert to numpy rgba array.
    from chimerax.graphics import qimage_to_numpy
    rgba = qimage_to_numpy(ti)
    p.end()
    return rgba

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, FloatArg, Float3Arg, BoolArg
    from chimerax.core.commands import ColormapArg, ColormapRangeArg, Color8Arg
    from chimerax.atomic import ResiduesArg
    desc = CmdDesc(
        required = [('residues', ResiduesArg),
                    ('score_name', StringArg)],
        keyword = [('scores_name', StringArg),
                   ('subtract_fit', StringArg),
                   ('range', ColormapRangeArg),
                   ('palette', ColormapArg),
                   ('no_data_color', Color8Arg),
                   ('height', FloatArg),
                   ('offset', Float3Arg),
                   ('on_top', BoolArg)],
        synopsis = 'Show color-coded residue labels for mutation scores'
    )
    register('mutationscores label', desc, mutation_scores_label, logger=logger)
