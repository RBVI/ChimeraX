Building Chimera2
=================

To get a working Chimera2 skeleton:

  1. In the top level directory, 'make install'.
  That will install the prerequisites and then build everything else.

  2. Later, just 'make install' in the code that you are working on. 

If available, pre-built versions of the prerequisites (i.e., binaries) are used 
to quickly bootstrap development.

The QT sample application is in src/qt.  To run it, type 'make run' in that
directory.  Later it will be in bin directory.


Changes from Chimera 1 Development Environment
----------------------------------------------

'$(build_root)' is now '$(build_prefix)'.

Prerequisite package installation is simplified:

    * Everything is installed in the 'build_prefix' directory
    * Package version numbers are local to the prerequisite's Makefile
    * The only make targets are 'install' and 'clean'
    * The 'foreign' directory is now 'prereqs'
    * The 'libs' and 'apps' directories are combined into 'src'
    * Source can optionally be unarchived in a directory in '$(tmpdir)'
    * 'clean' removes the unarchived directory
    * Python packages are installed in '$(PYTHON_SITE_PACKAGES)' instead of '$(PYSITEDIR)'
    * Python C/C++ libraries are installed in '$(pymoddir)'
