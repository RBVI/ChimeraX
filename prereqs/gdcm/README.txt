Use
---
GDCM, the Grassroots DICOM library is used by pydicom to open some dicom files, such as lossless 16-bit JPEG
encoded ones that Pillow cannot handle.  PyDicom will use GDCM if it finds it.

Distributions
-------------
GDCM seems to be poorly maintained making it difficult to obtain binary distributions for Python 3.7.

Mac and Linux
-------------
GDCM binaries for macOS and linux are taken from conda-forge

     https://anaconda.org/conda-forge/gdcm/files
     https://anaconda.org/conda-forge/json-c/files

A gdcm library depends on libjson-c which is not included so I merged in the json-c library by hand making
gdcm-2.8.9-mac-py37.tar.bz2 and gdcm-2.8.9-linux-py37.tar.bz2.

The macOS binary (GDCM-3.0.4) on the GDCM github site (https://github.com/malaterre/GDCM/releases) is compiled
for Python 2.7 so is not usable.

Windows
-------
On Windows I took a GDCM wheel compiled by the Github invesalius3 project.

   https://github.com/invesalius/invesalius3/wiki/Running-InVesalius-3-in-Windows
   
The Windows binary (GDCM-3.0.4) on the GDCM github site is compiled for Python 3.4 so is not usable.
