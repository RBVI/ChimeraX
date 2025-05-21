#!/usr/bin/env python3
import os
import subprocess

top = (
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
    .decode("utf-8")
    .strip()
)

prereq_directories = next(os.walk(os.path.join(top, "prereqs")))[1]
bundle_directories = next(os.walk(os.path.join(top, "src", "bundles")))[1]
srcapp_directories = next(os.walk(os.path.join(top, "src", "apps")))[1]

prereq_file_filters_file = os.path.join(top, ".github", "prereq-file-filters.yml")
bundle_file_filters_file = os.path.join(top, ".github", "bundle-file-filters.yml")
srcapp_file_filters_file = os.path.join(top, ".github", "srcapp-file-filters.yml")

with open(prereq_file_filters_file, "w") as f:
    for directory in prereq_directories:
        f.write(
            f'change_in_{directory.lower()}: &change_in_{directory.lower()}\n  - "prereqs/{directory}/**"\n'
        )
with open(bundle_file_filters_file, "w") as f:
    for directory in bundle_directories:
        f.write(
            f'change_in_{directory.lower()}: &change_in_{directory.lower()}\n  - "src/bundles/{directory}/**"\n'
        )
with open(srcapp_file_filters_file, "w") as f:
    for directory in srcapp_directories:
        f.write(
            f'change_in_{directory.lower()}: &change_in_{directory.lower()}\n  - "src/apps/{directory}/**"\n'
        )
