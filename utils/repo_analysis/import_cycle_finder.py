#!/usr/bin/env python3
import networkx
import re
import subprocess
import sys
import matplotlib.pyplot as plt

from collections import defaultdict

top_level_package_placeholder = "chimerax_init_func"

def get_bundle_name(bundle_path):
    # e.g. ./segger/seggerx/Segger/src/fit_dialog.py
    path_as_list = bundle_path.split('/')
    bundle_name = path_as_list[1]
    if bundle_name == "rotamer_libs":
        if path_as_list[2] == "Dunbrack":
            return "dunbrack_rotamer_lib"
        if path_as_list[2] == "Dynameomics":
            return "dynameomics_rotamer_lib"
        if path_as_list[2] == "Richardson":
            return "richardson_rotamer_lib"
    if bundle_name == "linux_support":
        return "linux"
    else:
        return bundle_name

def check_top_level(package):
    if package == "app_dirs":
        return True
    if package == "app_dirs_unversioned":
        return True
    if package == "app_lib_dir":
        return True
    if package == "app_data_dir":
        return True
    if package == "app_bin_dir":
        return True
    return False

def get_imported_bundle(import_line):
    no_whitespace = ' '.join(import_line.split())
    # It's either going to be 'from x import ...' or 'import x [as ...]', so in either case
    # split(' ')[1] is going to be chimerax, chimerax.core, chimerax.app_dirs, etc...
    first_word = no_whitespace.split(' ')[0]
    if first_word != 'from' and first_word != 'import':
        return []
    # Account for a malformed comment we don't want to process anyway
    if no_whitespace.startswith("import from"):
        return []
    if first_word == "from":
        package = no_whitespace.split(' ')[1]
        # No need to take relative, local imports into account
        if package == '.':
            return []
        if package == "chimerax":
            if no_whitespace == "import chimerax":
                return [top_level_package_placeholder]
            if no_whitespace.startswith("from chimerax import"):
                imports = no_whitespace.replace('from chimerax import ', '')
                imports = re.sub('#.*', '', imports)
                imports = re.sub(' as .*,', ',', imports)
                imports = re.sub(' as [_a-zA-Z]*$', '', imports)
                ' '.join(imports.split())
                if 'load_libarrays' in imports:
                    return ["arrays"]
                multiple_imports = imports.split(',')
                if len(multiple_imports) > 1:
                    multiple_imports = [x.strip() for x in multiple_imports]
                    if all([check_top_level(x) for x in multiple_imports]):
                        return [top_level_package_placeholder]
                    return sorted(list(set(multiple_imports)))
                else:
                    if check_top_level(imports):
                        return [top_level_package_placeholder]
                    return [imports]
        return [package.split('.')[1]]
    if first_word == "import":
        raw_package_list = no_whitespace.replace('import ', '')
        package_list = [x.strip() for x in raw_package_list.split(',')]
        chimerax_packages = list(filter(lambda x: 'chimerax' in x, package_list))
        final_packages = set()
        for raw_package in chimerax_packages:
            raw_package = re.sub(' as [_a-zA-Z]*$', '', raw_package)
            bundle_parts = raw_package.split('.')
            if len(bundle_parts) == 1:
                if bundle_parts[0] == 'chimerax':
                    final_packages.add(top_level_package_placeholder)
            else:
                if check_top_level(bundle_parts[1]):
                    final_packages.add(top_level_package_placeholder)
                else:
                    final_packages.add(bundle_parts[1])
        if len(final_packages) == 1:
            return list(final_packages)
        return sorted(list(final_packages))

def find_cycles(arg):
    imports = subprocess.run(["grep", "-R", "import"], capture_output = True)
    imports_from_sources = subprocess.run(["grep", "\\/src\\/"], input=imports.stdout, capture_output = True)
    chimerax_imports = subprocess.run(["grep", "chimerax"], input=imports_from_sources.stdout, capture_output = True)
    import_lines = chimerax_imports.stdout.decode().split('\n')
    bundles = {}
    for line in import_lines:
        # Get rid of any comments
        new_line = re.sub('#.*', '', line)
        if new_line == '':
            continue
        bundle, import_line = new_line.split(':')
        
        bundle_name = get_bundle_name(bundle)
        imported_bundles = get_imported_bundle(import_line)
  
        # TODO: Exclude list
        if bundle_name in ["core", "ui"]:
            continue
        if bundle_name not in bundles:
            bundles[bundle_name] = set()
        for entry in imported_bundles:
            if entry != bundle_name:
                bundles[bundle_name].add(entry)

    bundle_graph = networkx.DiGraph(bundles)
    if arg == "out_degree":
        nodes_by_out_degree = sorted([(node, bundle_graph.out_degree(node)) for node in bundle_graph], key = lambda x: x[1])
        print(nodes_by_out_degree)
    elif arg == "in_degree":
        nodes_by_out_degree = sorted([(node, bundle_graph.in_degree(node)) for node in bundle_graph], key = lambda x: x[1])
        print(nodes_by_in_degree)
    elif arg == "import_cycles":
        cycles = networkx.simple_cycles(bundle_graph)
        while cycles:
            print(' --> '.join(next(cycles)))
    else:
        print("Use one of {out_degree, in_degree, import_cycles} as an argument")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Use one of {out_degree, in_degree, import_cycles} as an argument")
        sys.exit(1)
    find_cycles(sys.argv[1])
