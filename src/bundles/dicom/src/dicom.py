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

from os.path import basename, commonprefix, dirname

from chimerax.core.models import Model

from chimerax.map import add_map_format
from chimerax.map.volume import open_grids, Volume
from chimerax.map_data import MapFileFormat

from .dicom_contours import DicomContours
from .dicom_format import find_dicom_series
from .dicom_grid import dicom_grids, dicom_grids_from_series

# https://stackoverflow.com/a/27361558/12208118
try:
    import gdcm # noqa
except Exception:
    _has_gdcm = False
else:
    _has_gdcm = True

def open_dicom(session, path, name = None, format = 'dicom', **kw):
    if isinstance(path, (str, list)):
        map_path = path         # Batched paths
    else:
        raise ValueError('open_dicom() requires path argument, got "%s"' % repr(path))

    # Locate all series in subdirectories
    series = find_dicom_series(map_path, log = session.logger, verbose = kw.get('verbose'))
    series = omit_16bit_lossless_jpeg(series, log = session.logger)

    # Open volume models for image series
    image_series = []
    contour_series = []
    extra_series = []
    for s in series:
        if s.has_image_data:
            image_series.append(s)
        elif s.dicom_class == 'RT Structure Set Storage':
            contour_series.append(s)
        else:
            extra_series.append(s)

    models, msg = dicom_volumes(session, image_series, **kw)

    # Open contour models for DICOM RT Structure Set series.
    if contour_series:
        cmodels, cmsg = dicom_contours(session, contour_series)
        models += cmodels
        msg += '\n' + cmsg
        # TODO: Associate contour models with image data they were derived from.

    # Warn about unrecognized series types.
    if extra_series:
        snames = ', '.join('%s (%s)' % (s.name, s.dicom_class) for s in extra_series)
        session.logger.warning('Can only handle images and contours, got %d other series types: %s'
                               % (len(extra_series), snames))

    gmodels = group_models(session, map_path, models)

    return gmodels, msg

def omit_16bit_lossless_jpeg(series, log):
    if _has_gdcm:
        return series    # PyDicom will use gdcm to read 16-bit lossless jpeg

    # Python Image Library cannot read 16-bit lossless jpeg.
    keep = []
    for s in series:
        if s.transfer_syntax == '1.2.840.10008.1.2.4.70' and s.attributes.get('BitsAllocated') == 16:
            if log:
                log.warning('Could not read DICOM %s because Python Image Library cannot read 16-bit lossless jpeg images.' % s.paths[0])
        else:
            keep.append(s)
    return keep
# -----------------------------------------------------------------------------
# Group into a four level hierarchy: directory, patient id, date, series.
# requires chimerax.core.models
def group_models(session, paths, models):
    if len(models) == 0:
        return []
    dname = basename(paths[0]) if len(paths) == 1 else basename(dirname(paths[0]))
    top = Model(dname, session)
    locations = []
    for m in models:
        s = model_series(m)
        if s is None:
            locations.append((m, ()))
        else:
            pid = s.attributes.get('PatientID', 'unknown')
            date = s.attributes.get('StudyDate', 'date unknown')
            locations.append((m, ('Patient %s' % pid, date)))

    leaf = {(): top}
    for m, gnames in locations:
        if gnames not in leaf:
            for i in range(len(gnames)):
                if gnames[:i + 1] not in leaf:
                    leaf[gnames[:i + 1]] = gm = Model(gnames[i], session)
                    leaf[gnames[:i]].add([gm])
        leaf[gnames].add([m])
    return [top]


# -----------------------------------------------------------------------------
#
def model_series(m):
    s = getattr(m, 'dicom_series', None)
    if s is None:
        # Look at child models for multi-channel and time-series.
        for c in m.child_models():
            s = getattr(c, 'dicom_series', None)
            if s:
                break
    return s


# -----------------------------------------------------------------------------
#
def dicom_volumes(session, series, **kw):
    grids = dicom_grids_from_series(series)
    models = []
    msg_lines = []
    sgrids = []
    for grid_group in grids:
        if isinstance(grid_group, (tuple, list)):
            # Handle multiple channels or time series
            gname = commonprefix([g.name for g in grid_group])
            if len(gname) == 0:
                gname = grid_group[0].name
            gmodels, gmsg = open_grids(session, grid_group, gname, **kw)
            models.extend(gmodels)
            msg_lines.append(gmsg)
        else:
            sgrids.append(grid_group)

    if sgrids:
        smodels, smsg = open_grids(session, sgrids, name, **kw) # noqa TODO: name undefined
        models.extend(smodels)
        msg_lines.append(smsg)

    for v in models:
        if isinstance(v, Volume):
            v.dicom_series = v.data.dicom_data.dicom_series
        else:
            for cv in v.child_models():
                if isinstance(cv, Volume):
                    cv.dicom_series = cv.data.dicom_data.dicom_series

    msg = '\n'.join(msg_lines)
    return models, msg

# -----------------------------------------------------------------------------
#
def dicom_contours(session, contour_series):
    models = [DicomContours(session, s) for s in contour_series]
    msg = 'Opened %d contour models' % len(models)
    return models, msg

# -----------------------------------------------------------------------------
#
class DICOMMapFormat(MapFileFormat):
    def __init__(self):
        MapFileFormat.__init__(self, 'DICOM image', 'dicom', ['dicom'], ['dcm'],
                               batch = True, allow_directory = True)

    @property
    def open_func(self):
        return self.open_dicom_grids

    def open_dicom_grids(self, paths, log = None, verbose = False):

        if isinstance(paths, str):
            paths = [paths]
        grids = dicom_grids(paths, log = log, verbose = verbose)
        return grids

# -----------------------------------------------------------------------------
#
def register_dicom_format(session):
    fmt = DICOMMapFormat()
    # Add map grid format reader
    add_map_format(session, fmt)
