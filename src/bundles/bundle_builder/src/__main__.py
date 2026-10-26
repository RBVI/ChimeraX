import argparse
import logging
import os
import sys

from chimerax.bundle_builder import xml_to_toml


def _get_builder(logger, path):
    if os.path.exists(os.path.join(os.path.abspath(path), "bundle_info.xml")):
        from chimerax.bundle_builder import BundleBuilder

        return BundleBuilder.from_path(logger, path)
    from chimerax.bundle_builder import BundleBuilderTOML

    return BundleBuilderTOML.from_path(logger, path)


def _build(path, debug=False, release=False):
    # Same as "devel build" but without starting ChimeraX, so wheels can be
    # built while other bundles are being installed
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    logger = logging.getLogger("bundle_builder")
    try:
        bb = _get_builder(logger, path)
        bb.make_wheel(debug=debug, release=release)
    except Exception:
        import traceback

        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="ChimeraX Bundle Builder")
    parser.add_argument(
        "-c",
        "--convert",
        type=str,
        help="Convert a bundle_info.xml to a pyproject.toml file",
    )
    parser.add_argument(
        "--dynamic-version",
        action="store_true",
        help="When converting, make the version number dynamic [e.g. relocate it to src/__init__.py]",
    )
    parser.add_argument(
        "-b",
        "--build",
        metavar="PATH",
        help="Build a wheel for the bundle in PATH without starting ChimeraX",
    )
    parser.add_argument(
        "--debug", action="store_true", help="When building, create a debug version"
    )
    parser.add_argument(
        "--release", action="store_true", help="When building, make a release build"
    )
    args = parser.parse_args()
    if args.convert:
        xml_to_toml(args.convert, dynamic_version=args.dynamic_version)
    if args.build:
        sys.exit(_build(args.build, debug=args.debug, release=args.release))
