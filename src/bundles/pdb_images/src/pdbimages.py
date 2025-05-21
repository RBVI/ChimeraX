# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

def pdbimages(session, directory = '.', subdirectories = True,
              width = 400, height = 400, supersample = 2,
              image_suffix = '.png', exclude = [],
              log_file = None):
    '''
    Assembly images command for batch rendering mmCIF assembly images.
    This is for the Protein DataBank to render images.

    Parameters
    ----------
    directory : string
        Find all files with suffix .cif in this directory and recursively in subdirectories.
    subdirectories : bool
        Whether to recursively search subdirectories for .cif files.
    width : integer
        Width in pixels of saved image.
    height : integer
        Height in pixels of saved image.
    supersample : integer
        Supersampling makes image N-fold larger than averages down to smooth object images.
    image_suffix : string
        File suffix for images, used to determine image format (e.g. .png, .jpg, .tif)
    exclude : list of strings
        File names to exclude from rendering (some large files cause graphics crashes).
    log_file : string
        Path to file to write log messages as each mmCIF file is rendered.
    '''
    s = session
    run(s, 'set bg white')
    run(s, 'set silhouette false')
    run(s, 'light soft')
    run(s, 'log settings warningDialog false errorDialog false')    # Avoid dialogs stopping batch rendering.

    mmcifs = cif_files(directory, subdirectories, exclude, image_suffix)

    from os.path import expanduser         # Tilde expansion
    log = open(expanduser(log_file), 'a') if log_file else NoLog()
    log.write('Rendering %d mmCIF files\n' % len(mmcifs))
    for f in mmcifs:
        log.write(f + '\n')
        log.flush()
        try:
            save_images(f, width, height, supersample, image_suffix, s)
        except Exception as e:
            log.write(str(e) + '\n')
    log.close()

class NoLog:
    def write(self, text):
        print(text)
    def flush(self):
        pass
    def close(self):
        pass

def cif_files(directory, subdirectories, exclude, image_suffix):

    from os import listdir, path
    directory = path.expanduser(directory)         # Tilde expansion
    files = listdir(directory)
    has_image = set(f.rsplit('_', maxsplit=1)[0]+'.cif' for f in files if f.endswith(image_suffix))
    mmcifs = [path.join(directory,f) for f in files
              if f.endswith('.cif') and not f in exclude and not f in has_image]
    # Include subdirectories
    for f in files:
        d = path.join(directory, f)
        if path.isdir(d):
            mmcifs.extend(cif_files(d, subdirectories, exclude, image_suffix))
    return mmcifs

def save_images(path, width, height, supersample, image_suffix, session):

    from chimerax.mmcif import open_mmcif
    mols, msg = open_mmcif(session, path)
    mol = mols[0]
    session.models.add([mol]) # Only use first model in nmr ensembles.

    run(session, 'ks cc')	# Color by chain

    from os.path import splitext
    image_prefix = splitext(path)[0]
    save_assembly_images(mol, width, height, supersample,
                         image_prefix, image_suffix, session)

    run(session, 'close')

def save_assembly_images(mol, width, height, supersample, image_prefix, image_suffix, session):
    s = session
    from chimerax.std_commands import sym
    for assembly in sym.pdb_assemblies(mol):
        run(s, 'sym #%s assembly %s' % (mol.id_string, assembly.id))
        run(s, 'view')          # Zoom to fit molecules
        image_file = '%s_%s%s' % (image_prefix, assembly.id, image_suffix)
        run(s, 'save "%s" width %d height %d supersample %d'
            % (image_file, width, height, supersample))

def run(session, cmd_text):
    from chimerax.core.commands import Command
    cmd = Command(session)
    cmd.run(cmd_text)

def register_command(logger):
    from chimerax.core.commands import CmdDesc, StringArg, SaveFolderNameArg, IntArg, \
        SaveFileNameArg, ListOf, register
    desc = CmdDesc(
        optional = [('directory', SaveFolderNameArg)],
        keyword = [('width', IntArg),
                   ('height', IntArg),
                   ('supersample', IntArg),
                   ('image_suffix', StringArg),
                   ('exclude', ListOf(StringArg)),
                   ('log_file', SaveFileNameArg)],
        synopsis = 'Render mmCIF assembly images')
    register('pdbimages', desc, pdbimages, logger=logger)

# To make a tiled array of images with filename labels:
# /opt/ImageMagick/bin/montage -label "%t" *.png -geometry "400x400+0+0" tiled.jpg
