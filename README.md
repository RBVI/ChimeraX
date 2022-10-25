<h1 align="center">UCSF ChimeraX Molecular Visualization</h1>
<div align="center">
  <a href='https://chimerax.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/chimerax/badge/?version=latest' alt='Documentation Status' />
  </a>
</div>

[ChimeraX](https://www.rbvi.ucsf.edu/chimerax/) is an application for visualizing and analyzing molecule structures such as proteins, RNA, DNA, lipids as well as gene sequences, electron microscopy maps, X-ray maps, 3D light microscopy and 3D medical imaging scans.  It is the successor of the [UCSF Chimera](https://www.cgl.ucsf.edu/chimera/) program.  [Example images](https://www.rbvi.ucsf.edu/chimerax/gallery.html) and [feature highlights](https://www.rbvi.ucsf.edu/chimerax/features.html) show a few of its capabilities, and [recipes](https://rbvi.github.io/chimerax-recipes/) show example command and Python API use.

## Installing ChimeraX

Nightly builds and semi-annual releases are available for Windows, macOS and Linux operating systems from the ChimeraX [downloads page](https://www.rbvi.ucsf.edu/chimerax/download.html).

## Platforms

ChimeraX runs on Windows, macOS and Linux and is free for academic and government use ([license](https://www.rbvi.ucsf.edu/chimerax/docs/license.html)).  Commercial use requires a [fee](https://www.rbvi.ucsf.edu/chimera/commercial_license.html) which supports development of the software.  It is about 80% Python 3 code and 20% C++ code.  The C++ is for speed critical parts.  It uses the Qt window toolkit.  

## Developing Plugins

The [ChimeraX Programming Manual](https://www.cgl.ucsf.edu/chimerax/docs/devel/index.html) describes the Python APIs available.

ChimeraX plugins are called bundles and can include reading new file formats, adding commands, adding GUI interfaces for new analysis tools, new visualization methods for data, new mouse modes using Python and C++ languages.  See the [ChimeraX Developer Tutorial](https://www.cgl.ucsf.edu/chimerax/docs/devel/tutorials/introduction.html) for information about creating bundles.

## Building ChimeraX

[Building ChimeraX](https://www.cgl.ucsf.edu/chimerax/docs/devel/building.html) can be challenging because it depends on more than 50 [third party libraries and packages](https://www.rbvi.ucsf.edu/chimerax/docs/devel/dependencies.html). We recommend using a prebuilt ChimeraX from the [downloads page](https://www.rbvi.ucsf.edu/chimerax/download.html) instead.  If you develop your own ChimeraX plugins they can be used with a prebuilt ChimeraX using the [toolshed install](http://www.rbvi.ucsf.edu/chimerax/docs/user/commands/toolshed.html#install) command.
