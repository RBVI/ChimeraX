Linux Build Environments
========================

Our policy is to only build for Linux LTS distributions that haven't
reached their End Of Life (EOL) and that the Qt pre-built binaries
run on.  But we concentrate on the two most common Linux variants in
the scientific community, Red Hat Enterprise Linux (and derivatives),
and Ubuntu.  We also make a "Generic Linux", which is currently based on
a version Debian.  The plan is to use a flatpak for the generic version.

All of the binary Linux ChimeraX bundles are built on a lower common
denominator (LCD) Linux.  The only platform-specific code is the Python
binary and dependencies (especially OpenSSL, so it gets security patches
from the vendor), and the AmberTools binaries that have a FORTRAN runtime
dependency.


Ubuntu End Of Life
-------------------

* Ubuntu 16.04: April 2024
* Ubuntu 18.04: April 2028
* Ubuntu 20.04: April 2030
* Ubuntu 22.04: April 2032


RHEL End Of Life
----------------

* RHEL 7: June 30, 2024
* RHEL 8: May 31, 2029
* RHEL 9: May 31, 2032


Qt 5 Requirements
-----------------

Qt 5 requires openSUSE 15.1, RHEL 7.6, or Ubuntu 18.04


Qt 6 Requirements
-----------------

Qt 6 requires RHEL 8.2, openSUSE 15.3, SUSE Server 15 SP2, or Ubuntu 20.04


ChimerX 1.3 and earlier
-----------------------

Uses Qt 5.  Generic Linux is Debian 8.  LCD Linux is RHEL/CentOS 7
with devtoolset-3.

ChimerX 1.4
-----------

Uses Qt 6.  Generic Linux is Debian 10.  LCD Linux is RHEL/CentOS 7
with devtoolset-9.

ChimerX 1.5
-----------

Uses Qt 6.  Generic Linux is Debian 10.  LCD Linux is RHEL/CentOS 8
with gcc-toolset-10.
