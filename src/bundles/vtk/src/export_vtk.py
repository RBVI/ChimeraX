#############################################################
#
# This module exports the Chimera Scene as a .vtk file for
# the input to tools using the Visualization Toolkit
#
# Author: Shawn Waldon
# Written as part of the SketchBio project, which is funded
# by the NIH (award number 50-P41-EB002025). November 2013.
#
# Ported from Chimera to ChimeraX by Tom Goddard, March 19, 2020
#

# custom exceptions for more descriptive error messages
# for when a point with a new array is added, but there are
# already points without that array
class PointWithUnknownArray(Exception):
    pass

# For when a point is added but does not have a data array
# that is currently defined, giving a point without an value
# for that array
class MissingDataArray(Exception):
    pass

# A class to contain and aggregate the data to be saved in the
# VTK file.
class DataToSave:
    def __init__(self):
        self.points = list()
        self.lines = list()
        self.triangles = list()
        self.arrays = dict()
        self.arrays['Normals'] = list()
    
    # Adds a point to the datastructure with the associated values for its
    # data arrays in the data input.  The position input should be a list or array
    # of position data, the data input should be a dict with keys being array names
    # and values being array values.  The only array that is treated specially is
    # 'Normals' which is its own input to this function and has a default (the input
    # may be left off).  Do not give a dict containing the key 'Normals' to the data
    # input to this function
    def addPoint(self, position, data, normal=(0,0,1)):
        self.points.append(position)
        index = len(self.points) - 1
        self.arrays['Normals'].append(normal);
        for key in data:
            if (key in self.arrays):
                self.arrays[key].append(data[key])
            else:
                self.arrays[key] = list([data[key]])
                if index > 0:
                    raise PointWithUnknownArray("Unknown array: %s" % key)
        for key in self.arrays:
            if len(self.arrays[key]) < len(self.points):
                raise MissingDataArray("Got no value for the %s data array" % key)

    # This function writes the data in the object to the given file in the VTK ascii
    # format.  This function contains all the logic about that format and provided that
    # the input data is already stored in this object, the file will be written correctly
    def writeToFile(self, vtkFile):
        # write out the header for the file
        vtkFile.write('# vtk DataFile Version 3.0\n')
        vtkFile.write('vtk file from Shawn Waldon\'s UCSF Chimera extension\n')
        vtkFile.write('ASCII\n')
        vtkFile.write('DATASET POLYDATA\n')
        # write out the points
        vtkFile.write('POINTS %d float\n' % len(self.points) )
        for point in self.points:
            vtkFile.write('%f %f %f\n' % (point[0], point[1], point[2]))
        # write out the lines
        if len(self.lines) > 0:
            vtkFile.write('\n\nLINES %d %d\n' % (len(self.lines), 3* len(self.lines)))
            for line in self.lines:
                vtkFile.write('2 %d %d\n' % (line[0], line[1]))
        # write out the triangles
        if len(self.triangles) > 0:
            vtkFile.write('\n\nPOLYGONS %d %d\n' % (len(self.triangles), 4 * len(self.triangles)))
            for tri in self.triangles:
                vtkFile.write('3 %d %d %d\n' % (tri[0], tri[1], tri[2]) )
        vtkFile.write('\n\nPOINT_DATA %d\n' % len(self.points))
        vtkFile.write('\n\nNORMALS %s %s\n' % ('Normals', 'float'))
        for norm in self.arrays['Normals']:
            vtkFile.write('%f %f %f\n' % (norm[0], norm[1], norm[2]))
        self.arrays.pop('Normals')
        # write the arrays
        vtkFile.write('\nFIELD FieldData %d\n' % len(self.arrays))
        for key in self.arrays:
            # write a vector array
            if isinstance(self.arrays[key][0], list):
                vtkFile.write('\n\nVECTORS %s %s\n' % (key, 'float'))
                for vec in self.arrays[key]:
                    vtkFile.write('%f %f %f\n' % (vec[0], vec[1], vec[2]))
            # write a scalar array, testing if it should be float or int
            # and writing the appropriate array
            elif not isinstance(self.arrays[key][0], str):
                isInt = isinstance(self.arrays[key][0], int)
                if isInt:
                    arrayType = 'int'
                else:
                    arrayType = 'float'
                vtkFile.write('\n\n%s 1 %d %s\n' % (key, len(self.arrays[key]), arrayType))
                count = 0
                for val in self.arrays[key]:
                    if (isInt):
                        vtkFile.write('%d\n' % val)
                    else:
                        vtkFile.write('%f\n' % val)
            # store that it is a string array
            else:
                vtkFile.write('\n%s 1 %d string\n' % (key, len(self.arrays[key])))
                for s in self.arrays[key]:
                    vtkFile.write('%s\n' % s)

# This function gets the list of open models from chimera
def getModels(session):
    from chimerax.atomic import Structure, MolecularSurface
    return session.models.list(type = (Structure, MolecularSurface))

# Parses the model and adds it to the datastructure
# Current data arrays created:
#    atomNum - the atom number within the model
#    atomType - the type of atom (string)  Just copying chimera's atom name specifier.
#    bFactor - the atom's B-Factor (something from the PDB data, no idea)
#    occupancy - the atom's occupancy (something from the PDB data, no idea)
#    modelNum - the model number (parameter)
#    chainPosition - the position along the chain (used for coloring in the
#                       Chimera command rainbow).  Value is fraction of chain length
#                       that is before this point (0 is N-terminus, 1 is C-terminus)
#    resType - a string array with the three letter description of the residue
#    resNum  - the residue number within the model (absolute residue id, not chain relative)
# Potential data arrays:
#   - removed due to VTK not understanding NaN in input files and no other good value for
#       invalid data:
#    kdHydrophobicity - the residue's Kite-Doolittle hydrophobicity (not available on all
#                           residues, defaults to 0.0 where no data)
def parseModel(m,modelNum,data):
    from chimerax.atomic import Structure, MolecularSurface
    if (isinstance(m,Structure)):
        offset = len(data.points)
        atoms = m.atoms
        residues = m.residues
        rcpos = residueChainPositions([m])
        aindex = {}
        for ai, atom in enumerate(atoms):
            aindex[atom] = ai
            pt = atom.scene_coord
            r = atom.residue
            arrays = { 'modelNum' : modelNum, 'atomNum' : ai,
                       'resType' : r.name, 'resNum' : residues.index(r),
                       'atomType' : atom.name, 'bFactor' : atom.bfactor,
                       'occupancy' : atom.occupancy, 'chainPosition': rcpos.get(r, 0.5) }
            #if atom.residue.kdHydrophobicity != None:
            #    arrays['kdHydrophobicity'] = atom.residue.kdHydrophobicity
            #else:
            #    arrays['kdHydrophobicity'] = 0.0 # float('NaN')
            data.addPoint(pt, arrays)
        for bond in m.bonds:
            a1, a2 = bond.atoms
            data.lines.append((aindex[a1] + offset, aindex[a2] + offset))
    elif isinstance(m, MolecularSurface):
        structures = m.atoms.unique_structures
        saindex = atomIndices(structures)
        srindex = residueIndices(structures)
        rcpos = residueChainPositions(structures)
        vertices = m.vertices
        triangles = m.triangles
        normals = m.normals
        v2a = m.vertex_to_atom_map()
        for pos in m.get_scene_positions():
            ptOffset = len(data.points)
            for i in range(0,len(vertices)):
                atomIdx = v2a[i]
                atom = m.atoms[atomIdx]
                r = atom.residue
                arrays = { 'modelNum' : modelNum,
                           'atomNum'  : saindex[atom],
                           'resType'  : r.name,
                           'resNum'   : srindex[r],
                           'atomType' : atom.name,
                           'bFactor'  : atom.bfactor,
                           'occupancy': atom.occupancy,
                           'chainPosition': rcpos.get(r,0.5),
                }
                # I would export hydrophobicity (and may in future versions) here,
                # but VTK doesn't read in NaN values
                v,n = (pos * vertices[i]), pos.transform_vector(normals[i])
                data.addPoint(v, arrays, normal = n)
            for tri in triangles:
                data.triangles.append((tri[0] + ptOffset, tri[1] + ptOffset, tri[2] + ptOffset))

def atomIndices(structures):
    saindex = {}
    for s in structures:
        saindex.update({a:i for i,a in enumerate(s.atoms)})
    return saindex

def residueIndices(structures):
    srindex = {}
    for s in structures:
        srindex.update({r:i for i,r in enumerate(s.residues)})
    return srindex

def residueChainPositions(structures):
    rcpos = {}  # Residue position in chain
    for s in structures:
        for c in s.chains:
            n = c.num_residues
            for ri, r in enumerate(c.residues):
                rcpos[r] = ri/n
    return rcpos

# parses chimera's datastructures and adds the data from each to the data object
def populate_data_object(models, data):
    mlist = list(models)
    for m in mlist:
        parseModel(m,mlist.index(m),data)

# writes the chimera scene to the file specified by path
def write_scene_as_vtk(session, path, models = None):
    data = DataToSave()
    if models is None:
        models = getModels(session)
    if len(models) == 0:
        from chimerax.core.errors import UserError
        raise UserError('Writing VTK no structure or molecular surface models specified.')
    populate_data_object(models, data)
    vtkFile = open(path, 'w')
    if len(data.points) > 0:
        data.writeToFile(vtkFile)
    vtkFile.close()

