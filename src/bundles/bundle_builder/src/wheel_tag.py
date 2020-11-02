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


def tag(pure, limited=False):
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
    limited : boolean
        True if Py_LIMITED_API is used to limit API

    Returns:
    --------
    str
        Dash-separated tag string, *e.g.*, **cp36-cp36m-win_amd64**
    """
    import sys
    from packaging import tags
    vi = sys.version_info
    if pure:
        # limit to current Python version, e.g., py38 instead of py3
        tag = tags.Tag(f"py{vi.major}{vi.minor}", "none", "any")
    else:
        # use most specific tag, e.g., manylinux2014_x86_64 instead of linux_x86_64
        if limited:
            abi = f"abi{vi.major}"
        for tag in tags.sys_tags():
            if not limited:
                break
            if tag.abi == abi:
                break
        else:
            raise RuntimeError("unable to find suitable tag")
    return tag


if "__main__" in __name__:
    import sys
    import getopt
    pure = False
    limited = False
    opts, args = getopt.getopt(sys.argv[1:], "pl")
    for opt, val in opts:
        if opt == "-p":
            pure = True
        elif opt == "-l":
            limited = True
    print(tag(pure, limited))
