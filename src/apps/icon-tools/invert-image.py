#!/bin/python3
# vi:set shiftwidth=4 expandtab:
#
# Add outline to existing image
#
import os
import sys
from PIL import Image, ImageOps

verbose = False


def invert_image(fn, inplace):
    im = Image.open(fn)
    if im.mode == 'RGBA':
        r, g, b, a = im.split()
        base = Image.merge('RGB', (r, g, b))
        tmp = ImageOps.invert(base)
        r2, g2, b2 = tmp.split()
        invert = Image.merge('RGBA', (r2, g2, b2, a))
    elif im.mode == 'LA':
        lu, a = im.split()
        tmp = ImageOps.invert(lu)
        lu2 = tmp.getchannel('L')
        invert = Image.merge('LA', (lu2, a))
    else:
        invert = ImageOps.invert(im)

    if inplace:
        new_fn = fn
    else:
        root, ext = os.path.splitext(fn)
        new_fn = root + "-invert" + ext
    if verbose:
        print("writing", new_fn)
    invert.save(new_fn)


def usage(file=sys.stderr):
    from textwrap import dedent
    print(dedent(f"""\
        usage: {sys.argv[0]} [-h|--help] [-i|--inplace] [-v|--verbose] image-file(s)
            --help      Show this help
            --inplace   Overwrite original file
            --verbose   Be chatty

            Invert the colors in an image.  By default, the image is to a "-invert"
            file unless --inplace is given.  The alpha mask is ignored.
        """), file=file)


def main():
    import getopt
    global verbose
    inplace = False

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hiv", [
            "help", "inplace", "verbose"])
    except getopt.GetoptError as err:
        print(err, file=sys.stderr)
        usage()
        raise SystemExit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage(sys.stdout)
            raise SystemExit(0)
        elif opt in ("-i", "--inplace"):
            inplace = True
        elif opt in ("-v", "--verbose"):
            verbose = True

    if len(args) == 0:
        print("error: missing file names", file=sys.stderr)
        raise SystemExit(1)

    for fn in args:
        if not os.path.exists(fn):
            print("warning: skipping missing file:", fn, file=sys.stderr)
            continue
        invert_image(fn, inplace)

    raise SystemExit(0)


if __name__ == "__main__":
    main()
