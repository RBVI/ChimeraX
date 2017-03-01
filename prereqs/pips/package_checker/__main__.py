#!/usr/bin/python3
# vim: set expandtab shiftwidth=4 softtabstop=4:

Comments = {
    "blockdiag": "required by Sphinx",
    "Cython": "required by pytables",
    "docutils": "required by Sphinx",
    "filelock": "",
    "flake8": "used for Python lint",
    "Jinja2": "required by Sphinx",
    "mccabe": "required by flake8",
    "numpydoc": "required for Sphinx",
    "pep8-naming": "required by flake8",
    "pyflakes": "required by flake8",
    "Pygments": "required by Sphinx",
    "python-dateutil": "required by pycollada",
    "sip": "used by PyQt",
    "Sphinx": "used by docs",
    "webcolors": "required by blockdiag",
}

def main():
    # Process arguments
    import getopt, sys
    try:
        opts, args = getopt.getopt(sys.argv[1:], "dhP:q")
    except getopt.GetoptError as e:
        print("Error: %s" % str(e), file=sys.stderr)
        print_help(sys.stderr)
        raise SystemExit(1)
    debug = False
    quiet = False
    packages = []
    for opt, val in opts:
        if opt == "-d":
            debug = True
        elif opt == "-P":
            packages.append(val)
        elif opt == "-q":
            quiet = True
        elif opt == "-h":
            print_help(sys.stdout)
            raise SystemExit(0)
    if len(args) == 0:
        print_help(sys.stderr)
        raise SystemExit(1)
    if not packages:
        from . import filter_collectors, report_ood
        ood_packages = out_of_date_packages()
        packages = [pkg["name"] for pkg in ood_packages]
        if not packages:
            if not quiet:
                print("No out-of-date packages found", file=sys.stderr)
            raise SystemExit(0)
        report_ood(ood_packages)

    # Collect import information from Python source files
    from . import collect
    collectors = []
    errors = []
    single = len(args) == 1
    for directory in args:
        cols, errs = collect(directory, single)
        collectors.extend(cols)
        errors.extend(errs)

    # Identify which files imported which packages
    from . import filter_collectors, report_importers
    from pkg_resources import get_distribution
    print("\nImported by:")
    for pkg in packages:
        import_names = list(get_distribution(pkg)._get_metadata('top_level.txt'))
        if len(import_names) == 0:
            import_names = [pkg]
        importers = []
        for mod in import_names:
            importers.extend(filter_collectors(collectors, mod))
        report_importers(importers, pkg, Comments.get(pkg, None))

    if errors and not quiet:
        print("\nErrors:")
        for msg in errors:
            print(" ", msg)

def print_help(f):
    import sys, os.path
    program = os.path.basename(sys.argv[0])
    print("Usage:", "python3 -m", __package__, "[-d]", "[-q]", "[-P package]",
          "directory...", file=f)
    print("         if -P is not used, check for out-of-date packages", file=f)
    print("         -P may be repeated multiple times to "
          "check several packages", file=f)
    print("   or:", "python3 -m", __package__, "[-h]", file=f)

def out_of_date_packages():
    # Ideally, there would be a pip API for getting this information
    # but since they clearly do not want to define the API, we use
    # pip as a command
    import sys, subprocess, json
    cp = subprocess.run([sys.executable, "-m", "pip", "list",
                         "--outdated", "--format=json"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
    if cp.returncode != 0:
        raise ValueError("pip terminated with exit status %d" % cp.returncode)
    pkgs = json.loads(cp.stdout.decode("utf-8"))
    return pkgs

if False:
    # Debug test against this source file
    def main():
        import sys
        c = Collector(sys.argv[0])
        print(c.module_names)

if False:
    # Debug test for out-of-date packages from pip
    def main():
        print(out_of_date_packages())

if __name__ == "__main__":
    main()
