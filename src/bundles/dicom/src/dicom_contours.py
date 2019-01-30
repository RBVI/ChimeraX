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

from chimerax.core.models import Model
class DicomContours(Model):
    def __init__(self, session, path):
        from pydicom import dcmread
        d = dcmread(path)

        desc = d.get('SeriesDescription', '')
        Model.__init__(self, 'Regions %s' % desc, session)

        cuid = d.get('SOPClassUID')
        if cuid is None:
            raise ValueError('DICOM file has no SOPClassUID, %s' % path)
        if cuid.name != 'RT Structure Set Storage':
            raise ValueError('DICOM file has SOPClassUID, %s, expected "RT Structure Set Storage", file %s'
                             % (cuid.name, path))

        def rgb_255(cs):
            return tuple(int(c) for c in cs)

        def xyz_list(xs):
            a = tuple(float(x) for x in xs)
            from numpy import array, float32
            xyz = array(a, float32).reshape(len(a)//3, 3)
            return xyz
        
        el = dicom_elements(d, {'StructureSetROISequence': {'ROINumber': int,
                                                            'ROIName': str},
                                'ROIContourSequence': {'ROIDisplayColor': rgb_255,
                                                       'ContourSequence': {'ContourGeometricType': str,
                                                                           'NumberOfContourPoints': int,
                                                                           'ContourData': xyz_list}}})
        regions = []
        for rs, rcs in zip(el['StructureSetROISequence'], el['ROIContourSequence']):
            r = ROIContourModel(session, rs['ROIName'], rs['ROINumber'], rcs['ROIDisplayColor'], rcs['ContourSequence'])
            regions.append(r)
        self.add(regions)

        session.models.add([self])

from chimerax.core.models import Surface
class ROIContourModel(Surface):
    def __init__(self, session, name, number, color, contour_info):
        Model.__init__(self, name, session)
        self.roi_number = number
        opacity = 255
        self.color = tuple(color) + (opacity,)
        va, ta = self._contour_lines(contour_info)
        self.set_geometry(va, None, ta)
        self.display_style = self.Mesh
        self.use_lighting = False
        
    def _contour_lines(self, contour_info):
        points = []
        triangles = []
        nv = 0
        from numpy import empty, int32, concatenate
        for ci in contour_info:
            ctype = ci['ContourGeometricType']
            if ctype != 'CLOSED_PLANAR':
                # TODO: handle other contour types
                continue
            np = ci['NumberOfContourPoints']
            pts = ci['ContourData']
            points.append(pts)
            n = len(pts)
            tri = empty((n,2), int32)
            tri[:,0] = tri[:,1] = range(n)
            tri[:,1] += 1
            tri[n-1,1] = 0
            tri += nv
            nv += n
            triangles.append(tri)
        from numpy import concatenate
        va = concatenate(points)
        ta = concatenate(triangles)
        return va, ta
        
def dicom_elements(data, fields):
    values = {}
    for name, v in fields.items():
        d = data.get(name)
        if d is None:
            raise ValueError('Did not find %s' % name)
        if isinstance(v, dict):
            values[name] = [dicom_elements(e, v) for e in d]
        else:
            values[name] = v(d)
    return values
            
