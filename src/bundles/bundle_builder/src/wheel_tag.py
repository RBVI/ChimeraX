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


def tag(pure, limited=None):
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
    limited : string
        Python version major[.minor[.micro]]

    Returns:
    --------
    str
        Dash-separated tag string, *e.g.*, **cp36-cp36m-win_amd64**
    """
    import sys
    from packaging import tags
    vi = sys.version_info
    if pure:
        if limited:
            version = ''.join(str(v) for v in limited.release[:2])
            tag = tags.Tag(f"py{version}", "none", "any")
        else:
            # savvy developers can handle default of all versions of Python 3
            tag = tags.Tag(f"py{vi.major}", "none", "any")
    else:
        target = None
        if sys.platform == "darwin":
            import os
            target = os.environ.get("MACOSX_DEPLOYMENT_TARGET", None)
            if target:
                target = f"_{target.replace('.', '_')}_"
        # use most specific tag, e.g., manylinux2014_x86_64 instead of linux_x86_64
        if limited:
            abi = f"abi{limited.major}"
            if limited.release < (3, 2):
                version = "32"
            else:
                version = ''.join(str(v) for v in limited.release[:2])
            interpreter = f"{tags.interpreter_name()}{version}"
        for tag in tags.sys_tags():
            if target and target not in tag.platform:
                continue
            if not limited:
                break
            if tag.abi == abi and tag.interpreter == interpreter:
                break
        else:
            raise RuntimeError("unable to find suitable tag")
    return tag


if "__main__" in __name__:
    import sys
    import getopt
    pure = False
    limited = False
    opts, args = getopt.getopt(sys.argv[1:], "pl:")
    for opt, val in opts:
        if opt == "-p":
            pure = True
        elif opt == "-l":
            from packaging.version import Version
            limited = Version(val)
    print(tag(pure, limited))
