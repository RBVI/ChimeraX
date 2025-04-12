# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===
__version__ = "1.4.0"
import importlib.metadata
import logging
import sys

from collections import defaultdict
from typing import Optional

from packaging.requirements import Requirement

from .bundle_builder import BundleBuilder  # noqa
from .bundle_builder_toml import Bundle as BundleBuilderTOML
from .metadata_templates import metadata_preamble, pure_wheel_platforms

log = logging.getLogger()

if sys.version_info < (3, 11, 0):
    import tomli as tomllib
else:
    import tomllib


# These are entry points for copying files into
# .dist-info directories of wheels when they are built
def copy_distinfo_file(cmd, basename, filename, binary=""):
    """Entry point to copy files into bundle .dist-info directory.

    File is copied as text if binary is '', and as binary if 'b'.
    """
    encoding = None if binary else "utf-8"
    try:
        with open(basename, "r" + binary, encoding=encoding) as fi:
            value = fi.read()
            import logging

            log = logging.getLogger()
            log.info("copying %s" % basename)
            if not cmd.dry_run:
                with open(filename, "w" + binary, encoding=encoding) as fo:
                    fo.write(value)
    except IOError:
        # Missing file is okay
        pass


def copy_distinfo_binary_file(cmd, basename, filename):
    copy_distinfo_file(cmd, basename, filename, binary="b")


package_toolshed_name_map: dict[str, str] = {
    "chimerax.add_charge": "ChimeraX-AddCharge",
    "chimerax.addh": "ChimeraX-AddH",
    "chimerax.alignment_algs": "ChimeraX-AlignmentAlgorithms",
    "chimerax.alignment_headers": "ChimeraX-AlignmentHdrs",
    "chimerax.alphafold": "ChimeraX-AlphaFold",
    "chimerax.altloc_explorer": "ChimeraX-AltlocExplorer",
    "chimerax.amber_info": "ChimeraX-AmberInfo",
    "chimerax.arrays": "ChimeraX-Arrays",
    "chimerax.atomsearch": "ChimeraX-AtomSearch",
    "chimerax.atomic": "ChimeraX-Atomic",
    "chimerax.atomic_lib": "ChimeraX-AtomicLibrary",
    "chimerax.axes_planes": "ChimeraX-AxesPlanes",
    "chimerax.basic_actions": "ChimeraX-BasicActions",
    "chimerax.bild": "ChimeraX-BILD",
    "chimerax.blastprotein": "ChimeraX-BlastProtein",
    "chimerax.bond_rot": "ChimeraX-BondRot",
    "chimerax.bug_reporter": "ChimeraX-BugReporter",
    "chimerax.build_structure": "ChimeraX-BuildStructure",
    "chimerax.bumps": "ChimeraX-Bumps",
    "chimerax.buttonpanel": "ChimeraX-ButtonPanel",
    "chimerax.cage_builder": "ChimeraX-CageBuilder",
    "chimerax.cellpack": "ChimeraX-CellPack",
    "chimerax.centroids": "ChimeraX-Centroids",
    "chimerax.chem_group": "ChimeraX-ChemGroup",
    "chimerax.clashes": "ChimeraX-Clashes",
    "chimerax.cmd_line": "ChimeraX-CommandLine",
    "chimerax.color_actions": "ChimeraX-ColorActions",
    "chimerax.color_globe": "ChimeraX-ColorGlobe",
    "chimerax.color_key": "ChimeraX-ColorKey",
    "chimerax.connect_structure": "ChimeraX-ConnectStructure",
    "chimerax.core": "ChimeraX-Core",
    "chimerax.core_formats": "ChimeraX-CoreFormats",
    "chimerax.coulombic": "ChimeraX-Coulombic",
    "chimerax.crosslinks": "ChimeraX-Crosslinks",
    "chimerax.crystal": "ChimeraX-Crystal",
    "chimerax.crystal_contacts": "ChimeraX-CrystalContacts",
    "chimerax.data_formats": "ChimeraX-DataFormats",
    "chimerax.dicom": "ChimeraX-Dicom",
    "chimerax.dist_monitor": "ChimeraX-DistMonitor",
    "chimerax.dssp": "ChimeraX-Dssp",
    "chimerax.dunbrack_rotamer_lib": "ChimeraX-RotamerLibsDunbrack",
    "chimerax.dynameomics_rotamer_lib": "ChimeraX-RotamerLibsDynameomics",
    "chimerax.emdb_sff": "ChimeraX-EMDB-SFF",
    "chimerax.exp_cmd": "ChimeraX-ExperimentalCommands",
    "chimerax.file_history": "ChimeraX-FileHistory",
    "chimerax.function_key": "ChimeraX-FunctionKey",
    "chimerax.geometry": "ChimeraX-Geometry",
    "chimerax.glft": "ChimeraX-gltf",
    "chimerax.graphics": "ChimeraX-Graphics",
    "chimerax.hbonds": "ChimeraX-Hbonds",
    "chimerax.help_viewer": "ChimeraX-Help",
    "chimerax.hkcage": "ChimeraX-HKCage",
    "chimerax.hubmap": "ChimeraX-HuBMAP",
    "chimerax.ihm": "ChimeraX-IHM",
    "chimerax.image_formats": "ChimeraX-ImageFormats",
    "chimerax.imod": "ChimeraX-IMOD",
    "chimerax.interfaces": "ChimeraX-Contacts",
    "chimerax.io": "ChimeraX-IO",
    "chimerax.items_inspection": "ChimeraX-ItemsInspection",
    "chimerax.label": "ChimeraX-Label",
    "chimerax.leap_motion": "ChimeraX-LeapMotion",
    "chimerax.linux": "ChimeraX-LinuxSupport",
    "chimerax.list_info": "ChimeraX-ListInfo",
    "chimerax.log": "ChimeraX-Log",
    "chimerax.looking_glass": "ChimeraX-LookingGlass",
    "chimerax.maestro": "ChimeraX-Maestro",
    "chimerax.map": "ChimeraX-Map",
    "chimerax.map_data": "ChimeraX-MapData",
    "chimerax.map_eraser": "ChimeraX-MapEraser",
    "chimerax.map_filter": "ChimeraX-MapFilter",
    "chimerax.map_fit": "ChimeraX-MapFit",
    "chimerax.map_series": "ChimeraX-MapSeries",
    "chimerax.markers": "ChimeraX-Markers",
    "chimerax.mask": "ChimeraX-Mask",
    "chimerax.match_maker": "ChimeraX-MatchMaker",
    "chimerax.md_crds": "ChimeraX-MDcrds",
    "chimerax.medical_toolbar": "ChimeraX-MedicalToolbar",
    "chimerax.meeting": "ChimeraX-Meeting",
    "chimerax.mlp": "ChimeraX-MLP",
    "chimerax.mmcif": "ChimeraX-mmCIF",
    "chimerax.mmtf": "ChimeraX-MMTF",
    "chimerax.model_panel": "ChimeraX-ModelPanel",
    "chimerax.model_series": "ChimeraX-ModelSeries",
    "chimerax.modeller": "ChimeraX-Modeller",
    "chimerax.mol2": "ChimeraX-Mol2",
    "chimerax.morph": "ChimeraX-Morph",
    "chimerax.mouse_modes": "ChimeraX-MouseModes",
    "chimerax.movie": "ChimeraX-Movie",
    "chimerax.neuron": "ChimeraX-Neuron",
    "chimerax.nih_presets": "ChimeraX-NIHPresets",
    "chimerax.nucleotides": "ChimeraX-Nucleotides",
    "chimerax.open_command": "ChimeraX-OpenCommand",
    "chimerax.pdb": "ChimeraX-PDB",
    "chimerax.pdb_bio": "ChimeraX-PDBBio",
    "chimerax.pdb_images": "ChimeraX-PDBImages",
    "chimerax.pdb_lib": "ChimeraX-PDBLibrary",
    "chimerax.pdb_matrices": "ChimeraX-PDBMatrices",
    "chimerax.phenix": "ChimeraX-Phenix",
    "chimerax.pick_blobs": "ChimeraX-PickBlobs",
    "chimerax.positions": "ChimeraX-Positions",
    "chimerax.preset_mgr": "ChimeraX-PresetMgr",
    "chimerax.pubchem": "ChimeraX-PubChem",
    "chimerax.read_pbonds": "ChimeraX-ReadPbonds",
    "chimerax.realsense": "ChimeraX-RealSense",
    "chimerax.registration": "ChimeraX-Registration",
    "chimerax.remote_control": "ChimeraX-RemoteControl",
    "chimerax.residue_fit": "ChimeraX-ResidueFit",
    "chimerax.rest_server": "ChimeraX-RestServer",
    "chimerax.richardson_rotamer_lib": "ChimeraX-RotamerLibsRichardson",
    "chimerax.rna_layout": "ChimeraX-RNALayout",
    "chimerax.rotamers": "ChimeraX-RotamerLibMgr",
    "chimerax.save_command": "ChimeraX-SaveCommand",
    "chimerax.scheme_mgr": "ChimeraX-SchemeMgr",
    "chimerax.sdf": "ChimeraX-SDF",
    "chimerax.segger": "ChimeraX-Segger",
    "chimerax.segment": "ChimeraX-Segment",
    "chimerax.sel_inspector": "ChimeraX-SelInspector",
    "chimerax.seqalign": "ChimeraX-Alignments",
    "chimerax.seqview": "ChimeraX-SeqView",
    "chimerax.shape": "ChimeraX-Shape",
    "chimerax.shell": "ChimeraX-Shell",
    "chimerax.shortcuts": "ChimeraX-Shortcuts",
    "chimerax.show_attr": "ChimeraX-ShowAttr",
    "chimerax.show_sequences": "ChimeraX-ShowSequences",
    "chimerax.sideview": "ChimeraX-SideView",
    "chimerax.signal_viewer": "ChimeraX-SignalViewer",
    "chimerax.sim_matrices": "ChimeraX-AlignmentMatrices",
    "chimerax.smiles": "ChimeraX-Smiles",
    "chimerax.smooth_lines": "ChimeraX-SmoothLines",
    "chimerax.spacenavigator": "ChimeraX-SpaceNavigator",
    "chimerax.speech": "ChimeraX-SpeechRecognition",
    "chimerax.std_commands": "ChimeraX-StdCommands",
    "chimerax.stl": "ChimeraX-STL",
    "chimerax.storm": "ChimeraX-Storm",
    "chimerax.struct_measure": "ChimeraX-StructMeasure",
    "chimerax.struts": "ChimeraX-Struts",
    "chimerax.surface": "ChimeraX-Surface",
    "chimerax.swap_res": "ChimeraX-SwapRes",
    "chimerax.swapaa": "ChimeraX-SwapAA",
    "chimerax.tape_measure": "ChimeraX-TapeMeasure",
    "chimerax.test": "ChimeraX-Test",
    "chimerax.toolbar": "ChimeraX-Toolbar",
    "chimerax.tug": "ChimeraX-Tug",
    "chimerax.ui": "ChimeraX-UI",
    "chimerax.uniprot": "ChimeraX-uniprot",
    "chimerax.unit_cell": "ChimeraX-UnitCell",
    "chimerax.viewdockx": "ChimeraX-ViewDockX",
    "chimerax.viperdb": "ChimeraX-VIPERdb",
    "chimerax.vive": "ChimeraX-Vive",
    "chimerax.volume_menu": "ChimeraX-VolumeMenu",
    "chimerax.vtk": "ChimeraX-VTK",
    "chimerax.wavefront_obj": "ChimeraX-WavefrontOBJ",
    "chimerax.webcam": "ChimeraX-WebCam",
    "chimerax.webservices": "ChimeraX-WebServices",
    "chimerax.zone": "ChimeraX-Zone",
    "cxwebservices": "ChimeraX-WebServices",
}

toolshed_package_name_map: dict[str, str] = {
    v: k for k, v in package_toolshed_name_map.items()
}


def _extract_license_from_classifiers(classifiers: list[str]) -> Optional[str]:
    license_classifier = None
    for classifier in classifiers:
        if classifier.startswith("License"):
            license_classifier = classifier
            classifiers.remove(classifier)
            break
    if license_classifier:
        license_classifier = (
            license_classifier.replace("License ::", "").rstrip().strip()
        )
    return license_classifier


def _extract_development_status_from_classifiers(
    classifiers: list[str],
) -> Optional[str]:
    development_classifier = None
    for classifier in classifiers:
        if classifier.startswith("Development Status"):
            development_classifier = classifier
            classifiers.remove(classifier)
            break
    return development_classifier


def _get_bundle_root(bundle_info):
    from lxml.etree import parse

    doc = parse(bundle_info)
    bi = doc.getroot()
    return bi


def _get_elements(e, tag):
    tagged_elements = list(e.iter(tag))
    # Mark element as used even for non-applicable platform
    elements = []
    for se in tagged_elements:
        elements.append(se)
    return elements


def _hack_build_deps_out_of_dependency_list(bundle_info):
    from lxml.etree import parse

    dependencies = []
    build_dependencies = []
    build_dependencies_as_modules = []
    bi = _get_bundle_root(bundle_info)
    deps = list(bi.iter("Dependencies"))
    if len(deps) > 0:
        deps = deps[0]
        for e in list(deps.iter("Dependency")):
            pkg = e.get("name", "")
            ver = e.get("version", "")
            build = e.get("build", False)
            req = "%s %s" % (pkg, ver)
            try:
                Requirement(req)
            except ValueError:
                raise ValueError("Bad version specifier (see PEP 440): %r" % req)
            dependencies.append(req)
            if build:
                build_dependencies.append(req)
                build_dependencies_as_modules.append(
                    toolshed_package_name_map.get(pkg, pkg)
                )
    if _any_compiled_extensions(bundle_info):
        core_dep = [dep for dep in dependencies if dep.startswith("ChimeraX-Core")]
        if not core_dep:
            # We are in the arrays bundle which deceptively omits its dependency on the
            # core because XML bundle builder includes core headers automatically. TOML
            # bundle builder on the other hand is a T crossing and I dotting kind of guy.
            core_dep = ["ChimeraX-Core ~=1.0"]
        build_dependencies.append(core_dep[0])
    return dependencies, build_dependencies, build_dependencies_as_modules


def _any_compiled_extensions(bundle_info) -> bool:
    bi = _get_bundle_root(bundle_info)
    for _ in list(bi.iter("CModule")):
        return True
    for _ in list(bi.iter("CLibrary")):
        return True
    return False


def _any_extension_uses_numpy(bundle_info) -> bool:
    # Go through the extensions to see if any of them use numpy so we can add it to the
    # list of build dependencies
    bi = _get_bundle_root(bundle_info)
    for e in list(bi.iter("CModule")):
        if e.get("usesNumpy", False):
            return True
    for e in list(bi.iter("CLibrary")):
        if e.get("usesNumpy", False):
            return True
    return False


def _path_to_source_tree_location(path: str, dir: bool = False) -> str:
    if dir:
        return "/".join(["src", path])
    return "/".join(["src", *path.split("/")[:-1]])


def _extract_data_files(bundle_info) -> dict[str, list[str]]:
    # Once again we are unable to use the bundle builder functions;
    # TOML bundle builder is slightly smarter than XML bundle builder.
    # There is no need to specify the name of an extension or library
    # as module data. We must go through extensions and make sure to
    # filter them out of package data, and then fix up the paths.
    bi = _get_bundle_root(bundle_info)
    dfiles_block = list(bi.iter("DataFiles"))
    data_files = defaultdict(list)
    if dfiles_block:
        dfiles_block = dfiles_block[0]
        module_and_library_names = []
        for e in list(bi.iter("CModule")):
            module_and_library_names.append(e.get("name"))
        for e in list(bi.iter("CLibrary")):
            module_and_library_names.append(e.get("name"))
        for e in list(dfiles_block.iter("DataFile")):
            if len(BundleBuilder._get_element_text(e).split(".")) > 1:
                file, extension = BundleBuilder._get_element_text(e).split(".", 1)
            else:
                file = BundleBuilder._get_element_text(e)
                extension = ""
            if file in module_and_library_names:
                continue
            elif extension in ["so", "dylib", "dll"]:
                # These don't have to be listed as data files, they get
                # included automatically.
                continue
            else:
                fixed_path = _path_to_source_tree_location(file)

                if extension == "":
                    if len(file.split("/")) > 1:
                        maybe_file = file.split("/")[-1]
                        if maybe_file == "*":
                            data_files[fixed_path].append("*")
                else:
                    if len(file.split("/")) > 1:
                        # Are we dealing with a path?
                        maybe_file = file.split("/")[-1]
                        data_files[fixed_path].append(".".join([maybe_file, extension]))
                    else:
                        data_files[fixed_path].append(".".join([file, extension]))
    return data_files


def _extract_extra_files(bundle_info) -> dict[str, list[str]]:
    bi = _get_bundle_root(bundle_info)
    efiles_block = list(bi.iter("ExtraFiles"))
    extra_files = defaultdict(list)
    if efiles_block:
        efiles_block = efiles_block[0]
        for e in list(efiles_block.iter("ExtraFile")):
            source = e.get("source")
            file = BundleBuilder._get_element_text(e)
            fixed_path = _path_to_source_tree_location(file)
            extra_files[fixed_path + "/"].append(source)
        for e in list(efiles_block.iter("ExtraDir")):
            source = e.get("source") + "/*"
            file = BundleBuilder._get_element_text(e)
            fixed_path = _path_to_source_tree_location(file, dir=True)
            extra_files[fixed_path].append(source)
    return extra_files


def _extract_tools(bundle_info) -> list[dict[str, list[str]]]:
    bi = _get_bundle_root(bundle_info)
    tools = []
    classifiers_block = list(bi.iter("Classifiers"))
    if classifiers_block:
        classifiers_block = classifiers_block[0]
        for e in list(classifiers_block.iter("ChimeraXClassifier")):
            classifier = BundleBuilder._get_element_text(e)
            if classifier.startswith("Tool"):
                data = classifier.split("::")
                _, name, category, description = data
                tool_dict = {}
                tool_dict["name"] = name.strip().rstrip()
                tool_dict["category"] = category.strip().rstrip()
                tool_dict["description"] = description.strip().rstrip()
                tools.append(tool_dict)
    return tools


def _extract_commands(bundle_info) -> list[dict[str, str]]:
    bi = _get_bundle_root(bundle_info)
    commands = []
    classifiers_block = list(bi.iter("Classifiers"))
    if classifiers_block:
        classifiers_block = classifiers_block[0]
        for e in list(classifiers_block.iter("ChimeraXClassifier")):
            classifier = BundleBuilder._get_element_text(e)
            if classifier.startswith("Command"):
                data = classifier.split("::")
                _, name, category, description = data
                command_dict = {}
                command_dict["name"] = name.strip().rstrip()
                command_dict["category"] = category.strip().rstrip()
                command_dict["description"] = description.strip().rstrip()
                commands.append(command_dict)
    return commands


def _extract_providers(bundle_info) -> list[dict[str, list[str]]]:
    bi = _get_bundle_root(bundle_info)
    providers_blocks = list(bi.iter("Providers"))
    providers = []
    if providers_blocks:
        for provider_block in providers_blocks:
            for provider in provider_block:
                # Who knows why but mmcif has a ghost provider that needs
                # ignoring
                if not provider.attrib:
                    continue
                provider_dict = {}
                provider_dict["manager"] = provider_block.get("manager")
                provider_dict |= provider.attrib
                providers.append(provider_dict)
    return providers


def _extract_managers(bundle_info) -> list[dict[str, list[str]]]:
    bi = _get_bundle_root(bundle_info)
    managers_blocks = list(bi.iter("Managers"))
    managers = []
    if managers_blocks:
        for manager_block in managers_blocks:
            for manager in manager_block:
                if not manager.attrib:
                    continue
                manager_dict = {}
                manager_dict["manager"] = manager_block.get("manager")
                manager_dict |= manager.attrib
                managers.append(manager_dict)
    return managers


def _extract_c_extensions(
    bundle_info, build_dependencies_as_module_names, extension_type
) -> list[dict[str, list[str]]]:
    windows_platforms = ["win32", "windows"]
    mac_platforms = ["darwin", "macos", "mac"]
    linux_platforms = ["linux"]
    bi = _get_bundle_root(bundle_info)
    c_modules = []
    for e in list(bi.iter(extension_type)):
        module_dict = defaultdict(list)
        module_windows_dict = defaultdict(list)
        module_mac_dict = defaultdict(list)
        module_linux_dict = defaultdict(list)

        module_dict["name"] = e.get("name")
        if e.get("static", False):
            module_dict["static"] = e.get("static")
        module_windows_dict["name"] = ".".join([e.get("name"), "win32"])
        module_mac_dict["name"] = ".".join([e.get("name"), "mac"])
        module_linux_dict["name"] = ".".join([e.get("name"), "linux"])
        for source_file in _get_elements(e, "SourceFile"):
            module_dict["sources"].append(source_file.text)
        for include_dir in _get_elements(e, "IncludeDir"):
            module_dict["include-dirs"].append(include_dir.text)
        for library_dir in _get_elements(e, "LibraryDir"):
            module_dict["library-dirs"].append(library_dir.text)
        for library in _get_elements(e, "Library"):
            if library.get("platform") in windows_platforms:
                module_windows_dict["libraries"].append(library.text)
            elif library.get("platform") in linux_platforms:
                module_linux_dict["libraries"].append(library.text)
            elif library.get("platform") in mac_platforms:
                module_mac_dict["libraries"].append(library.text)
            else:
                module_dict["libraries"].append(library.text)
        for framework in _get_elements(e, "Framework"):
            module_mac_dict["frameworks"].append(framework.text)
        for compile_arg in _get_elements(e, "CompileArgument"):
            if compile_arg.get("platform") in windows_platforms:
                module_windows_dict["extra-compile-args"].append(compile_arg.text)
            elif compile_arg.get("platform") in linux_platforms:
                module_linux_dict["extra-compile-args"].append(compile_arg.text)
            elif compile_arg.get("platform") in mac_platforms:
                module_mac_dict["extra-compile-args"].append(compile_arg.text)
            else:
                module_dict["extra-compile-args"].append(compile_arg.text)
        for link_arg in _get_elements(e, "LinkArgument"):
            if link_arg.get("platform") in windows_platforms:
                module_windows_dict["extra-link-args"].append(link_arg.text)
            elif link_arg.get("platform") in linux_platforms:
                module_linux_dict["extra-link-args"].append(link_arg.text)
            elif link_arg.get("platform") in mac_platforms:
                module_mac_dict["extra-link-args"].append(link_arg.text)
            else:
                module_dict["extra-link-args"].append(link_arg.text)
        for define in _get_elements(e, "Define"):
            if define.get("platform") in windows_platforms:
                module_windows_dict["define-macros"].append(define.text)
            elif define.get("platform") in linux_platforms:
                module_linux_dict["define-macros"].append(define.text)
            elif define.get("platform") in mac_platforms:
                module_mac_dict["define-macros"].append(define.text)
            else:
                module_dict["define-macros"].append(define.text)
        module_dict["include-modules"].extend(build_dependencies_as_module_names)
        module_dict["include-modules"].append("chimerax.core")
        if e.get("usesNumpy", False):
            module_dict["include-modules"].append("numpy")
        module_dict["library-modules"].extend(build_dependencies_as_module_names)
        module_dict["library-modules"].append("chimerax.core")
        c_modules.append(module_dict)
        if len(module_windows_dict.keys()) > 1:
            c_modules.append(module_windows_dict)
        if len(module_mac_dict.keys()) > 1:
            c_modules.append(module_mac_dict)
        if len(module_linux_dict.keys()) > 1:
            c_modules.append(module_linux_dict)
    return c_modules


def _extract_selectors(bundle_info) -> list[dict[str, str]]:
    bi = _get_bundle_root(bundle_info)
    selectors = []
    classifiers_block = list(bi.iter("Classifiers"))
    if classifiers_block:
        classifiers_block = classifiers_block[0]
        for e in list(classifiers_block.iter("ChimeraXClassifier")):
            classifier = BundleBuilder._get_element_text(e)
            if classifier.startswith("Selector") or classifier.startswith(
                "ChimeraX :: Selector"
            ):
                if classifier.startswith("ChimeraX :: Selector"):
                    classifier = classifier.replace("ChimeraX :: Selector", "Selector")
                data = classifier.split("::")
                _, name, description = data
                selector_dict = {}
                selector_dict["name"] = name.strip().rstrip()
                selector_dict["description"] = description.strip().rstrip()
                selectors.append(selector_dict)
    return selectors


def xml_to_toml(
    bundle_info, dynamic_version=False, write: bool = False, quiet: bool = False
) -> Optional[str]:
    bundle = BundleBuilder(logger=log, bundle_xml=bundle_info)
    python_classifiers = bundle.python_classifiers
    license_classifier = _extract_license_from_classifiers(python_classifiers)
    if not license_classifier:
        raise ValueError("No license PythonClassifier found in input bundle XML file")
    development_status_classifier = _extract_development_status_from_classifiers(
        python_classifiers
    )
    # Now filter out all the classifiers that automatically get added to bundles since
    # we're just going to add them again in TOML bundle builder
    automatically_written_metadata = metadata_preamble.split("\n")
    automatically_written_platforms = pure_wheel_platforms.split("\n")
    extra_classifiers = []
    for classifier in python_classifiers:
        if not (
            classifier in automatically_written_metadata
            or classifier in automatically_written_platforms
        ):
            extra_classifiers.append(classifier)
    if development_status_classifier:
        extra_classifiers.append(development_status_classifier)

    dependencies, build_dependencies, build_dependencies_as_module_names = (
        _hack_build_deps_out_of_dependency_list(bundle_info)
    )

    if _any_extension_uses_numpy(bundle_info):
        reqs = importlib.metadata.requires("ChimeraX-BundleBuilder")
        if reqs:
            for dep in reqs:
                if dep.startswith("numpy"):
                    build_dependencies.append(dep.replace(" ", ""))
                    break

    # Set the build backend to require this version of bundle builder or greater.
    # Bundle builder will pull in the appropriate version of setuptools when
    # it is installed.

    if build_dependencies:
        toml = (
            """[build-system]\nrequires = [\n  "ChimeraX-BundleBuilder>=%s",\n  "%s",\n]\n"""
            % (
                __version__,
                '",\n  "'.join(build_dependencies),
            )
        )
    else:
        toml = (
            """[build-system]\nrequires = [\n  "ChimeraX-BundleBuilder>=%s",\n]\n"""
            % __version__
        )
    toml += """build-backend = "chimerax.bundle_builder.cx_pep517"\n\n"""
    toml += """[project]\nname = "%s"\n""" % bundle.name
    if not dynamic_version:
        toml += 'version = "%s"\n' % bundle.version
    toml += 'license = { text = "%s" }\n' % license_classifier
    toml += 'authors = [{ name = "%s", email = "%s" }]\n' % (
        bundle.author,
        bundle.email,
    )
    toml += 'description = "%s"\n' % bundle.synopsis
    if not dependencies:
        pass
    elif len(dependencies) == 1:
        toml += 'dependencies = ["%s"]\n' % dependencies[0]
    else:
        toml += 'dependencies = [\n  "%s",\n]\n' % '",\n  "'.join(dependencies)
    toml += 'dynamic = ["classifiers", "requires-python"'
    if dynamic_version:
        toml += ', "version"'
    toml += "]\n\n"

    toml += "[project.readme]\n"
    toml += 'content-type = "text"\n'
    toml += 'text = """%s"""\n\n' % bundle.description

    toml += "[project.urls]\n"
    toml += 'Home = "%s"\n\n' % bundle.url

    if dynamic_version:
        toml += (
            """[tool.setuptools.dynamic]\nversion = { attr = "src.__version__" }\n\n"""
        )
    toml += (
        """[tool.chimerax]\nmin-session-version = %i\nmax-session-version = %i\n"""
        % (int(bundle.min_session), int(bundle.max_session))
    )
    if bundle.supersedes:
        toml += """supersedes = "%s"\n""" % bundle.supersedes
    if bundle.package:
        pkg = bundle.package
        bundle_base_name, module_name, dist_info_name = (
            BundleBuilderTOML.format_module_name(bundle.name)
        )
        if module_name != pkg:
            toml += """module-name-override = "%s"\n""" % bundle.package.replace(
                "chimerax.", ""
            )
    if bundle.limited_api:
        toml += """limited-api = "%s"\n""" % bundle.limited_api
    # We never write pure python in TOML bundle builder because you can tell whether
    # or not it's pure pyuthon by the presence of CModules, CLibraries, and/or executables
    if bundle.custom_init:
        toml += """custom-init = %s\n""" % bundle.custom_init
    if bundle.categories:
        toml += """categories = [\"%s\"]\n""" % '", "'.join(bundle.categories)
    if extra_classifiers:
        if len(extra_classifiers) == 1:
            toml += """classifiers = ["%s"]\n""" % extra_classifiers[0]
        else:
            toml += "classifiers = [\n"
            for classifier in extra_classifiers:
                toml += '    "%s",\n' % classifier
            toml += "]\n"
    toml += "\n"

    datafiles = _extract_data_files(bundle_info)
    if datafiles:
        toml += "[tool.chimerax.package-data]\n"
        for path, files in datafiles.items():
            toml += '"%s" = [\n' % path
            for file in files:
                toml += '  "%s",\n' % file
            toml += "]\n"
        toml += "\n"

    extrafiles = _extract_extra_files(bundle_info)
    if extrafiles:
        toml += "[tool.chimerax.extra-files]\n"
        for path, files in extrafiles.items():
            toml += '"%s" = [\n' % path
            for file in files:
                toml += '  "%s",\n' % file
            toml += "]\n"
        toml += "\n"

    tools = _extract_tools(bundle_info)
    for tool in tools:
        if len(tool["name"].split(" ")) > 1:
            toml += '[tool.chimerax.tool."%s"]\n' % tool["name"]
        else:
            toml += "[tool.chimerax.tool.%s]\n" % tool["name"]
        for key, value in tool.items():
            if key == "name":
                continue
            else:
                toml += '%s = "%s"\n' % (key, value)
        toml += "\n"

    commands = _extract_commands(bundle_info)
    for command in commands:
        if len(command["name"].split(" ")) > 1 or command["name"].startswith("~"):
            toml += '[tool.chimerax.command."%s"]\n' % command["name"]
        else:
            toml += "[tool.chimerax.command.%s]\n" % command["name"]
        for key, value in command.items():
            if key == "name":
                continue
            else:
                toml += '%s = "%s"\n' % (key, value)
        toml += "\n"

    managers = _extract_managers(bundle_info)
    for manager in managers:
        toml += '[tool.chimerax.manager."%s"]\n' % manager["name"]
        for key, value in manager.items():
            if key == "manager" or key == "name":
                continue
            else:
                if value in ["true", "false"]:
                    toml += "%s = %s\n" % (key.replace("_", "-"), value)
                else:
                    toml += '%s = "%s"\n' % (key, value)
        toml += "\n"

    providers = _extract_providers(bundle_info)
    for provider in providers:
        toml += '[[tool.chimerax.provider."%s"]]\n' % provider["manager"]
        for key, value in provider.items():
            if key == "manager":
                continue
            else:
                if value in ["true", "false"]:
                    toml += "%s = %s\n" % (key.replace("_", "-"), value)
                else:
                    toml += '%s = "%s"\n' % (key.replace("_", "-"), value.encode('unicode_escape').decode('ascii'))
        toml += "\n"

    c_extensions = _extract_c_extensions(
        bundle_info, build_dependencies_as_module_names, "CModule"
    )
    c_libs = _extract_c_extensions(
        bundle_info, build_dependencies_as_module_names, "CLibrary"
    )

    for extension in c_extensions:
        toml += "[tool.chimerax.extension.%s]\n" % extension["name"]
        if bundle.name == "ChimeraX-Arrays":
            # Just as arrays doesn't declare its dependency on the core, it also doesn't
            # declare its dependency on numpy for its module. We must add it manually.
            extension["include-modules"].append("numpy")
        # Some bundles kind of abuse the fact that XML bunlde builder doesn't always
        # put the library down in the lib dir, but TOML bundle builder ALWAYS will,
        # so add these paths if they don't exist
        if (
            not extension["name"].endswith(".win32")
            and not extension["name"].endswith(".mac")
            and not extension["name"].endswith(".linux")
        ):
            if "src/include" not in extension["include-dirs"]:
                extension["include-dirs"].append("src/include")
            if "src/lib" not in extension["library-dirs"]:
                extension["library-dirs"].append("src/lib")
        for key, value in extension.items():
            if key == "name":
                continue
            else:
                if value in ["true", "false"]:
                    toml += "%s = %s\n" % (key.replace("_", "-"), value)
                else:
                    toml += "%s = [\n" % key
                    for val in value:
                        toml += '  "%s",\n' % val
                    toml += "]\n"
        toml += "\n"

    for extension in c_libs:
        toml += "[tool.chimerax.library.%s]\n" % extension["name"]
        for key, value in extension.items():
            if key == "name":
                continue
            else:
                if value in ["true", "false"]:
                    toml += "%s = %s\n" % (key.replace("_", "-"), value)
                else:
                    toml += "%s = [\n" % key
                    for val in value:
                        toml += '  "%s",\n' % val
                    toml += "]\n"
        toml += "\n"

    selectors = _extract_selectors(bundle_info)
    for selector in selectors:
        if len(selector["name"].split(" ")) > 1 or bundle.name == "ChimeraX-Atomic":
            toml += '[tool.chimerax.selector."%s"]\n' % selector["name"]
        else:
            toml += "[tool.chimerax.selector.%s]\n" % selector["name"]
        toml += 'description = "%s"\n' % selector["description"]
        toml += "\n"

    # """
    # categories = ["Sequence"]
    # module-name-override = "%s"
    # classifiers = ["Development Status :: 2 - Pre-Alpha"]
    # """ % bundle.package

    # """
    # [chimerax.tool."Blast Protein"]
    # category = "Sequence"
    # description = "Search PDB/NR/AlphaFold using BLAST"
    #
    # [chimerax.command.blastprotein]
    # category = "Sequence"
    # description = "Search PDB/NR/AlphaFold using BLAST"
    #
    # [chimerax.command.blastpdb]
    # category = "Sequence"
    # description = "Search PDB/NR/AlphaFold using BLAST"
    #
    # [chimerax.command."blastprotein pull"]
    # category = "Sequence"
    # description = "Get results for a finished BlastProtein job"
    # """

    if not quiet:
        if dynamic_version:
            print(
                "\nYou enabled dynamic versioning; you must add the following to your top level __init__.py:"
            )
            print('__version__ = "%s"' % bundle.version)
            print("This must be the first uncommented line in the file.\n")

        print(toml)
    return toml
