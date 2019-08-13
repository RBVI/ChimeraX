# vim: set expandtab shiftwidth=4 softtabstop=4:

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


class ImageFormat:
    def __init__(self, name, suffixes, pil_name):
        self.name = name
        self.suffixes = suffixes
        self.pil_name = pil_name


_formats = [
    ('png', ['png'], 'PNG'),
    ('jpeg', ['jpg', 'jpeg'], 'JPEG'),
    ('tiff', ['tif', 'tiff'], 'TIFF'),
    ('gif', ['gif'], 'GIF'),
    ('ppm', ['ppm'], 'PPM'),
    ('bmp', ['bmp'], 'BMP'),
]
default_format = 'png'
image_formats = [ImageFormat(name, suffixes, pil_name)
                 for name, suffixes, pil_name in _formats]


def save_image(session, path, format_name, width=None, height=None,
               supersample=3, pixel_size=None, transparent_background=False, quality=95):
    '''
    Save an image of the current graphics window contents.
    '''
    from .errors import UserError, LimitationError
    has_graphics = session.main_view.render is not None
    if not has_graphics:
        raise LimitationError("Unable to save images because OpenGL rendering is not available")
    from os.path import expanduser, dirname, exists, splitext
    path = expanduser(path)         # Tilde expansion
    dir = dirname(path)
    if dir and not exists(dir):
        raise UserError('Directory "%s" does not exist' % dir)

    if pixel_size is not None:
        if width is not None or height is not None:
            raise UserError('Cannot specify width or height if pixel_size is given')
        v = session.main_view
        b = v.drawing_bounds()
        if b is None:
            raise UserError('Cannot specify use pixel_size option when nothing is shown')
        psize = v.pixel_size(b.center())
        if psize > 0 and pixel_size > 0:
            f = psize / pixel_size
            w, h = v.window_size
            from math import ceil
            width, height = int(ceil(f * w)), int(ceil(f * h))
        else:
            raise UserError('Pixel size option (%g) and screen pixel size (%g) must be positive'
                            % (pixel_size, psize))

    fmt = None
    if format_name is not None:
        for f in image_formats:
            if f.name == format_name:
                fmt = f
        if fmt is None:
            from .errors import UserError
            raise UserError('Unknown image file format "%s"' % format_name)

    suffix = splitext(path)[1][1:].casefold()
    if suffix == '':
        if fmt is None:
            fmt = default_format
            path += '.' + default_format.suffixes[0]
        else:
            path += '.' + fmt.suffixes[0]
    elif fmt is None:
        for f in image_formats:
            if suffix in f.suffixes:
                fmt = f
        if fmt is None:
            from .errors import UserError
            raise UserError('Unknown image file suffix "%s"' % suffix)

    from .session import standard_metadata
    std_metadata = standard_metadata()
    metadata = {}
    if fmt.name == 'png':
        metadata['optimize'] = True
        # if dpi is not None:
        #     metadata['dpi'] = (dpi, dpi)
        from PIL import PngImagePlugin
        pnginfo = PngImagePlugin.PngInfo()
        # tags are from <https://www.w3.org/TR/PNG/#11textinfo>

        def add_text(keyword, value):
            try:
                b = value.encode('latin-1')
            except UnicodeEncodeError:
                pnginfo.add_itxt(keyword, value)
            else:
                pnginfo.add_text(keyword, b)
        # add_text('Title', description)
        add_text('Creation Time', std_metadata['created'])
        add_text('Software', std_metadata['generator'])
        add_text('Author', std_metadata['creator'])
        add_text('Copy' 'right', std_metadata['dateCopyrighted'])
        metadata['pnginfo'] = pnginfo
    elif fmt.name == 'tiff':
        # metadata['compression'] = 'lzw:2'
        # metadata['description'] = description
        metadata['software'] = std_metadata['generator']
        # TIFF dates are YYYY:MM:DD HH:MM:SS (local timezone)
        import datetime as dt
        metadata['date_time'] = dt.datetime.now().strftime('%Y:%m:%d %H:%M:%S')
        metadata['artist'] = std_metadata['creator']
        # TIFF copy right is ASCII, so no Unicode symbols
        cp = std_metadata['dateCopyrighted']
        if cp[0] == '\N{COPYRIGHT SIGN}':
            cp = 'Copy' 'right' + cp[1:]
        metadata['copy' 'right'] = cp
        # if units == 'pixels':
        #     dpi = None
        # elif units in ('points', 'inches'):
        #     metadata['resolution unit'] = 'inch'
        #     metadata['x resolution'] = dpi
        #     metadata['y resolution'] = dpi
        # elif units in ('millimeters', 'centimeters'):
        #     adjust = convert['centimeters'] / convert['inches']
        #     dpcm = dpi * adjust
        #     metadata['resolution unit'] = 'cm'
        #     metadata['x resolution'] = dpcm
        #     metadata['y resolution'] = dpcm
    elif fmt.name == 'jpeg':
        metadata['quality'] = quality
        # if dpi is not None:
        #     # PIL's jpeg_encoder requires integer dpi values
        #     metadata['dpi'] = (int(dpi), int(dpi))
        # TODO: create exif with metadata using piexif package?
        # metadata['exif'] = exif

    view = session.main_view
    i = view.image(width, height, supersample=supersample,
                   transparent_background=transparent_background)
    if i is not None:
        i.save(path, fmt.pil_name, **metadata)
    else:
        session.logger.warning("Unable to save image of size %d by %d." % (width, height))


def register_image_save(session):
    from .io import register_format
    for format in image_formats:
        register_format("%s image" % format.name,
                        category='Image',
                        extensions=['.%s' % s for s in format.suffixes],
                        nicknames=[format.name.casefold()],
                        export_func=save_image)

    # Register save command keywords for images
    from .commands import PositiveIntArg, FloatArg, BoolArg, Bounded, IntArg
    from .commands.cli import add_keyword_arguments
    save_image_args = [
        ('width', PositiveIntArg),
        ('height', PositiveIntArg),
        ('supersample', PositiveIntArg),
        ('pixel_size', FloatArg),
        ('transparent_background', BoolArg),
        ('quality', Bounded(IntArg, min=0, max=100)),
    ]
    add_keyword_arguments('save', dict(save_image_args))

    # Register save image subcommand
    from .commands import CmdDesc, register, SaveFileNameArg
    from .commands.save import SaveFileFormatsArg, save
    desc = CmdDesc(
        required=[('filename', SaveFileNameArg)],
        keyword=[('format', SaveFileFormatsArg('Image'))] + save_image_args,
        synopsis='save image'
    )
    register('save image', desc, save, logger=session.logger)

    
def register_image_save_options_gui(save_dialog):
    '''
    Image save gui options are registered in the ui module instead of when the
    format is registered because the ui does not exist when the format is registered.
    '''
    #
    # Options for Save File dialog.
    #
    from chimerax.ui import SaveOptionsGUI
    class ImageSaveOptionsGUI(SaveOptionsGUI):

        SUPERSAMPLE_OPTIONS = (("None", None),
                               ("2x2", 2),
                               ("3x3", 3),
                               ("4x4", 4))

        def __init__(self, image_format):
            self._image_format = image_format
            SaveOptionsGUI.__init__(self)
            
        @property
        def format_name(self):
            return self._image_format.name
        
        def make_ui(self, parent):
            from PyQt5.QtWidgets import QFrame, QGridLayout, QComboBox, QLabel, QHBoxLayout, \
                QLineEdit, QCheckBox
            container = QFrame(parent)
            layout = QGridLayout(container)
            layout.setContentsMargins(2, 0, 0, 0)
            row = 0

            size_frame = QFrame(container)
            size_layout = QHBoxLayout(size_frame)
            size_layout.setContentsMargins(0, 0, 0, 0)
            self._width = w = QLineEdit(size_frame)
            new_width = int(0.4 * w.sizeHint().width())
            w.setFixedWidth(new_width)
            w.textEdited.connect(self._width_changed)
            x = QLabel(size_frame)
            x.setText("x")
            self._height = h = QLineEdit(size_frame)
            h.setFixedWidth(new_width)
            h.textEdited.connect(self._height_changed)
            
            from PyQt5.QtCore import Qt
            size_layout.addWidget(self._width, Qt.AlignRight)
            size_layout.addWidget(x, Qt.AlignHCenter)
            size_layout.addWidget(self._height, Qt.AlignLeft)
            size_frame.setLayout(size_layout)
            size_label = QLabel(container)
            size_label.setText("Size:")
            layout.addWidget(size_label, row, 0, Qt.AlignRight | Qt.AlignVCenter)
            layout.addWidget(size_frame, row, 1, Qt.AlignLeft)
            row += 1

            self._keep_aspect = ka = QCheckBox('preserve aspect', container)
            ka.setChecked(True)
            ka.stateChanged.connect(self._aspect_changed)
            layout.addWidget(ka, row, 1, Qt.AlignLeft)
            row += 1

            ss_label = QLabel(container)
            ss_label.setText("Supersample:")
            supersamples = QComboBox(container)
            supersamples.addItems([o[0] for o in self.SUPERSAMPLE_OPTIONS])
            layout.addWidget(ss_label, row, 0, Qt.AlignRight | Qt.AlignVCenter)
            layout.addWidget(supersamples, row, 1, Qt.AlignLeft)
            self._supersample = supersamples

            container.setLayout(layout)
            return container

        def _width_changed(self):
            if self._keep_aspect.isChecked():
                w,h,iw,ih = self._sizes()
                if w > 0 and iw is not None:
                    self._height.setText('%.0f' % ((iw/w) * h))
        
        def _height_changed(self):
            if self._keep_aspect.isChecked():
                w,h,iw,ih = self._sizes()
                if h > 0 and ih is not None:
                    self._width.setText('%.0f' % ((ih/h) * w))

        def _sizes(self):
            gw = self._session.ui.main_window.graphics_window
            w, h = gw.width(), gw.height()
            try:
                iw = int(self._width.text())
            except ValueError:
                iw = None
            try:
                ih = int(self._height.text())
            except ValueError:
                ih = None
            return w, h, iw, ih

        def _aspect_changed(self, state):
            if self._keep_aspect.isChecked():
                w,h,iw,ih = self._sizes()
                if iw != w:
                    self._width_changed()
                else:
                    self._height_changed()
        
        def save(self, session, filename):

            # Add file suffix if needed
            import os.path
            ext = os.path.splitext(filename)[1]
            suf = self._image_format.suffixes
            if ext[1:] not in suf:
                filename += '.' + suf[0]

            # Get image width and height
            try:
                w = int(self._width.text())
                h = int(self._height.text())
            except ValueError:
                from chimerax.core.errors import UserError
                raise UserError("width/height must be integers")
            if w <= 0 or h <= 0:
                from chimerax.core.errors import UserError
                raise UserError("width/height must be positive integers")

            # Get supersampling
            ss = self.SUPERSAMPLE_OPTIONS[self._supersample.currentIndex()][1]

            # Run image save command
            from chimerax.core.commands import run, quote_if_necessary
            cmd = "save image %s width %g height %g" % (quote_if_necessary(filename), w, h)
            if ss is not None:
                cmd += " supersample %g" % ss
            run(session, cmd)

        def update(self, session, save_dialog):
            self._session = session
            gw = session.ui.main_window.graphics_window
            w, h = gw.width(), gw.height()
            self._width.setText(str(w))
            self._height.setText(str(h))

        def wildcard(self):
            f = self._image_format
            suf = ' '.join("*.%s" % e for e in f.suffixes)
            wildcard = "%s image file (%s)" % (f.name.upper(), suf)
            return wildcard

    for fmt in image_formats:
        save_dialog.register(ImageSaveOptionsGUI(fmt))
