# vi: set expandtab shiftwidth=4 softtabstop=4:

def pdbimages(session, directory = '.', subdirectories = True,
              width = 400, height = 400, supersample = 2,
              image_suffix = '.png', exclude = ['1m4x.cif'],
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
    run(s, 'log warningDialog false errorDialog false')    # Avoid dialogs stopping batch rendering.

    mmcifs = cif_files(directory, subdirectories, exclude, image_suffix)

    from os.path import expanduser         # Tilde expansion
    log = open(expanduser(log_file), 'a') if log_file else NoLog()
    log.write('Rendering %d mmCIF files\n' % len(mmcifs))
    for f in mmcifs:
        log.write(f + '\n')
        log.flush()
        try:
            save_images(f, width, height, supersample, image_suffix, s)
        except Error as e:
            log.write(str(e) + '\n')
    log.close()

class NoLog:
    def write(self, text):
        pass
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

    s = session
    m = s.models
    mols = m.open(path)
    m.close(mols[1:]) # Only use first model in nmr ensembles.
    mol = mols[0]

    run(s, 'ks cc')	# Color by chain

    from os.path import splitext
    image_prefix = splitext(path)[0]
    save_assembly_images(mol, width, height, supersample,
                         image_prefix, image_suffix, session)

    run(s, 'close')

def save_assembly_images(mol, width, height, supersample, image_prefix, image_suffix, session):
    s = session
    from . import sym
    for assembly in sym.pdb_assemblies(mol):
        run(s, 'sym #%s assembly %s' % (mol.id_string(), assembly.id))
        run(s, 'view')          # Zoom to fit molecules
        image_file = '%s_%s%s' % (image_prefix, assembly.id, image_suffix)
        run(s, 'save "%s" width %d height %d supersample %d'
            % (image_file, width, height, supersample))

def run(session, cmd_text):
    from . import Command
    cmd = Command(session, cmd_text, final=True)
    cmd.execute()

def register_command(session):
    from . import CmdDesc, StringArg, IntArg, ListOf, register
    desc = CmdDesc(
        optional = [('directory', StringArg)],
        keyword = [('width', IntArg),
                   ('height', IntArg),
                   ('supersample', IntArg),
                   ('image_suffix', StringArg),
                   ('exclude', ListOf(StringArg)),
                   ('log_file', StringArg)],
        synopsis = 'Render mmCIF assembly images')
    register('pdbimages', desc, pdbimages)

# To make a tiled array of images with filename labels:
# /opt/ImageMagick/bin/montage -label "%t" *.png -geometry "400x400+0+0" tiled.jpg

