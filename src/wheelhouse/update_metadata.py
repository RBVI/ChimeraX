# vi: set expandtab shiftwidth=4 softtabstop=4:
"""update_metadata:

Reads all the wheel-format file in a directory and
reates a JSON file of the metadata information in the wheels.
"""

def main():
    import getopt, sys
    opts, args = getopt.getopt(sys.argv[1:], "vo:d:")
    if args:
        usage(sys.stderr)
        raise SystemExit(1)
    output = "METADATA.json"
    directory = "."
    for opt, val in opts:
        if opt == "-d":
            directory = val
        elif opt == "-o":
            output = val
        elif opt == "-v":
            import logging
            logging.basicConfig(level=logging.INFO)
    update_metadata(directory, output)


def usage(f):
    print("Usage: %s [-v] [-o output_file] [-d package_dir]" % _program,
                                    file=f)
    print("\t-v            \tPrint verbose messages", file=f)
    print("\t-o output_file\tMetadata output file name", file=f)
    print("\t-d package_dir\tPackage directory path", file=f)


def update_metadata(directory, output):
    import os, os.path, logging
    logging.info("Processing directory %s" % directory)
    packages = []
    for filename in os.listdir(directory):
        logging.debug("Checking file %s" % filename)
        if os.path.splitext(filename)[1] != ".whl":
            continue
        path = os.path.join(directory, filename)
        logging.info("Processing file %s" % path)
        try:
            md = _metadata(path)
            md["wheel"] = filename
            packages.append(md)
        except Exception as e:
            logging.error("%s: %s" % (path, str(e)))
    path = os.path.join(directory, output)
    logging.info("Writing output file %s" % path)
    _print_packages(packages, path)
    logging.info("Finished directory %s" % directory)


def _metadata(path):
    from wheel.install import WheelFile
    try:
        wf = WheelFile(path)
    except Exception as e:
        print("%s: %s" % (path, str(e)))
    metadata_file = "%s/metadata.json" % wf.distinfo_name
    with wf.zipfile.open(metadata_file) as mdf:
        import json
        md = json.loads(mdf.read().decode(encoding="UTF-8"))
        md["modified"] = _modtime(path)
        return md


def _modtime(path):
    from os.path import getmtime
    t = getmtime(path)
    from time import strftime, gmtime
    return strftime("%Y-%m-%d %H:%M:%S", gmtime(t))


def _print_packages(packages, output):
    import json
    try:
        if callable(output.write):
            json.dump(packages, f)
    except AttributeError:
        with open(output, "w") as f:
            json.dump(packages, f)


if __name__ == "__main__":
    main()
