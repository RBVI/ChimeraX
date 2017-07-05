# vim: set expandtab shiftwidth=4 softtabstop=4:

"""
Script to generate the wheel compatibility tag for the invoking
version of Python.

Basic Usage
-----------

*python* -m chimerax.wheel_tag
    Print tag for pure Python packages, *e.g.*, **cp36-cp36m-win_amd64**
*python* -m chimerax.wheel_tag -p
    Print tag for Python packages with compiled code, *e.g.*, **py3-none-any**
"""

def tag(pure):
    """Return the tag part of a wheel filename for this version of Python.

    https://www.python.org/dev/peps/pep-0491/#file-name-convention
    describes the filename convention for wheels.  'tag' returns the
      {python tag}-{abi tag}-{platform tag}
    part of the filename that is compatible with the executing
    version of Python.

    Parameters:
    -----------
    pure : boolean
        Whether the bundle only contains Python code (no C/C++)

    Returns:
    --------
    str
        Dash-separated tag string, *e.g.*, **cp36-cp36m-win_amd64**
    """
    # Code below is taken from wheel==0.29 bdist_wheel.py
    from wheel.pep425tags import get_impl_ver
    impl_ver = get_impl_ver()
    if pure:
        impl = "py" + impl_ver[0]
        abi = "none"
        platform = "any"
    else:
        from wheel.pep425tags import get_abbr_impl, get_abi_tag
        impl = get_abbr_impl() + impl_ver
        # get_abi_tag generates warning messages
        import warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        abi = get_abi_tag()
        from distutils.util import get_platform
        platform = get_platform()

    def fix_name(name):
        return name.replace('-', '_').replace('.', '_')
    return "%s-%s-%s" % (fix_name(impl), fix_name(abi), fix_name(platform))

if "__main__" in __name__:
    import sys, getopt
    pure = False
    opts, args = getopt.getopt(sys.argv[1:], "p")
    for opt, val in opts:
        if opt == "-p":
            pure = True
    print(tag(pure))
