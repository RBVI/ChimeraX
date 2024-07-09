#!/bin/python3
# vi:set shiftwidth=4 expandtab:
#
# Add outline to existing image
#
import os
import sys
from PIL import Image, ImageChops, ImageColor

verbose = False


def add_outline(fn, color, inplace=False):
    im = Image.open(fn)
    # alpha_composite only works for RGBA images
    if im.mode == 'RGBA':
        grayscale = False
        blend = Image.new('RGBA', size=im.size, color=color)
    elif im.mode == 'LA':
        grayscale = True
        blend = Image.new('RGBA', size=im.size, color=color)
        im = im.convert('RGBA')

    alpha = im.getchannel('A')
    original = alpha.copy()

    for offset in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        offset_image = Image.new('L', im.size)
        offset_image.paste(original, offset)
        alpha = ImageChops.add(alpha, offset_image, .5)
    outline_alpha = ImageChops.subtract(alpha, original)

    blend.putalpha(outline_alpha)
    im.alpha_composite(blend)

    if grayscale:
        im = im.convert('LA')
    if not inplace:
        root, ext = os.path.splitext(fn)
        fn = root + "-outline" + ext
    if verbose:
        print("writing", fn)
    im.save(fn)


def usage(file=sys.stderr):
    from textwrap import dedent
    print(dedent(f"""\
        usage: {sys.argv[0]} [-c|--color outline] [-h|--help] [-i|--inplace] [-v|--verbose] image-file(s)
            --color     Color name or specification (default black)
            --help      Show this help
            --inplace   Overwrite original file
            --verbose   Be chatty

            Add colored outline to raster image files.  By default, the image
            is to a -outline file unless --inplace is given.  For the various
            was to specify the color, see:
            https://pillow.readthedocs.io/en/stable/reference/ImageColor.html.
        """), file=file)


def main():
    import getopt
    global verbose
    color = (0, 0, 0)  # black
    inplace = False

    try:
        opts, args = getopt.getopt(sys.argv[1:], "c:hiv", [
            "color=", "help", "inplace", "verbose"])
    except getopt.GetoptError as err:
        print(err, file=sys.stderr)
        usage()
        raise SystemExit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage(sys.stdout)
            raise SystemExit(0)
        if opt in ("-c", "--color"):
            try:
                color = ImageColor.getrgb(arg)
            except ValueError:
                print("error: unknown color", file=sys.stderr)
                raise SystemExit(1)
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
        add_outline(fn, color, inplace)

    raise SystemExit(0)


if __name__ == "__main__":
    main()
