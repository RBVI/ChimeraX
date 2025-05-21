import argparse
from chimerax.bundle_builder import xml_to_toml

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
    args = parser.parse_args()
    if args.convert:
        xml_to_toml(args.convert, dynamic_version=args.dynamic_version)
