# vim: set et ts=4 sts=4:
# === UCSF ChimeraX Copyright ====
# Copyright 2022 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ====
import sys
import toml

from packaging.requirements import Requirement
from typing import Optional

package_toolshed_name_map: dict[str, str] = {
    'chimerax.add_charge': 'ChimeraX-AddCharge'
    , 'chimerax.addh': 'ChimeraX-AddH'
    , 'chimerax.alignment_algs': 'ChimeraX-AlignmentAlgorithms'
    , 'chimerax.alignment_headers': 'ChimeraX-AlignmentHdrs'
    , 'chimerax.alphafold': 'ChimeraX-AlphaFold'
    , 'chimerax.altloc_explorer': 'ChimeraX-AltlocExplorer'
    , 'chimerax.amber_info': 'ChimeraX-AmberInfo'
    , 'chimerax.arrays': 'ChimeraX-Arrays'
    , 'chimerax.atomsearch': 'ChimeraX-AtomSearch'
    , 'chimerax.atomic': 'ChimeraX-Atomic'
    , 'chimerax.atomic_lib': 'ChimeraX-AtomicLibrary'
    , 'chimerax.axes_planes': 'ChimeraX-AxesPlanes'
    , 'chimerax.basic_actions': 'ChimeraX-BasicActions'
    , 'chimerax.bild': 'ChimeraX-BILD'
    , 'chimerax.blastprotein': 'ChimeraX-BlastProtein'
    , 'chimerax.bond_rot': 'ChimeraX-BondRot'
    , 'chimerax.bug_reporter': 'ChimeraX-BugReporter'
    , 'chimerax.build_structure': 'ChimeraX-BuildStructure'
    , 'chimerax.bumps': 'ChimeraX-Bumps'
    , 'chimerax.buttonpanel': 'ChimeraX-ButtonPanel'
    , 'chimerax.cage_builder': 'ChimeraX-CageBuilder'
    , 'chimerax.cellpack': 'ChimeraX-CellPack'
    , 'chimerax.centroids': 'ChimeraX-Centroids'
    , 'chimerax.chem_group': 'ChimeraX-ChemGroup'
    , 'chimerax.clashes': 'ChimeraX-Clashes'
    , 'chimerax.cmd_line': 'ChimeraX-CommandLine'
    , 'chimerax.color_actions': 'ChimeraX-ColorActions'
    , 'chimerax.color_globe': 'ChimeraX-ColorGlobe'
    , 'chimerax.color_key': 'ChimeraX-ColorKey'
    , 'chimerax.connect_structure': 'ChimeraX-ConnectStructure'
    , 'chimerax.core': 'ChimeraX-Core'
    , 'chimerax.core_formats': 'ChimeraX-CoreFormats'
    , 'chimerax.coulombic': 'ChimeraX-Coulombic'
    , 'chimerax.crosslinks': 'ChimeraX-Crosslinks'
    , 'chimerax.crystal': 'ChimeraX-Crystal'
    , 'chimerax.crystal_contacts': 'ChimeraX-CrystalContacts'
    , 'chimerax.data_formats': 'ChimeraX-DataFormats'
    , 'chimerax.dicom': 'ChimeraX-Dicom'
    , 'chimerax.dist_monitor': 'ChimeraX-DistMonitor'
    , 'chimerax.dssp': 'ChimeraX-Dssp'
    , 'chimerax.dunbrack_rotamer_lib': 'ChimeraX-RotamerLibsDunbrack'
    , 'chimerax.dynameomics_rotamer_lib': 'ChimeraX-RotamerLibsDynameomics'
    , 'chimerax.emdb_sff': 'ChimeraX-EMDB-SFF'
    , 'chimerax.exp_cmd': 'ChimeraX-ExperimentalCommands'
    , 'chimerax.file_history': 'ChimeraX-FileHistory'
    , 'chimerax.function_key': 'ChimeraX-FunctionKey'
    , 'chimerax.geometry': 'ChimeraX-Geometry'
    , 'chimerax.glft': 'ChimeraX-gltf'
    , 'chimerax.graphics': 'ChimeraX-Graphics'
    , 'chimerax.hbonds': 'ChimeraX-Hbonds'
    , 'chimerax.help_viewer': 'ChimeraX-Help'
    , 'chimerax.hkcage': 'ChimeraX-HKCage'
    , 'chimerax.hubmap': 'ChimeraX-HuBMAP'
    , 'chimerax.ihm': 'ChimeraX-IHM'
    , 'chimerax.image_formats': 'ChimeraX-ImageFormats'
    , 'chimerax.imod': 'ChimeraX-IMOD'
    , 'chimerax.interfaces': 'ChimeraX-Contacts'
    , 'chimerax.io': 'ChimeraX-IO'
    , 'chimerax.items_inspection': 'ChimeraX-ItemsInspection'
    , 'chimerax.label': 'ChimeraX-Label'
    , 'chimerax.leap_motion': 'ChimeraX-LeapMotion'
    , 'chimerax.linux': 'ChimeraX-LinuxSupport'
    , 'chimerax.list_info': 'ChimeraX-ListInfo'
    , 'chimerax.log': 'ChimeraX-Log'
    , 'chimerax.looking_glass': 'ChimeraX-LookingGlass'
    , 'chimerax.maestro': 'ChimeraX-Maestro'
    , 'chimerax.map': 'ChimeraX-Map'
    , 'chimerax.map_data': 'ChimeraX-MapData'
    , 'chimerax.map_eraser': 'ChimeraX-MapEraser'
    , 'chimerax.map_filter': 'ChimeraX-MapFilter'
    , 'chimerax.map_fit': 'ChimeraX-MapFit'
    , 'chimerax.map_series': 'ChimeraX-MapSeries'
    , 'chimerax.markers': 'ChimeraX-Markers'
    , 'chimerax.mask': 'ChimeraX-Mask'
    , 'chimerax.match_maker': 'ChimeraX-MatchMaker'
    , 'chimerax.md_crds': 'ChimeraX-MDcrds'
    , 'chimerax.medical_toolbar': 'ChimeraX-MedicalToolbar'
    , 'chimerax.meeting': 'ChimeraX-Meeting'
    , 'chimerax.mlp': 'ChimeraX-MLP'
    , 'chimerax.mmcif': 'ChimeraX-mmCIF'
    , 'chimerax.mmtf': 'ChimeraX-MMTF'
    , 'chimerax.model_panel': 'ChimeraX-ModelPanel'
    , 'chimerax.model_series': 'ChimeraX-ModelSeries'
    , 'chimerax.modeller': 'ChimeraX-Modeller'
    , 'chimerax.mol2': 'ChimeraX-Mol2'
    , 'chimerax.morph': 'ChimeraX-Morph'
    , 'chimerax.mouse_modes': 'ChimeraX-MouseModes'
    , 'chimerax.movie': 'ChimeraX-Movie'
    , 'chimerax.neuron': 'ChimeraX-Neuron'
    , 'chimerax.nih_presets': 'ChimeraX-NIHPresets'
    , 'chimerax.nucleotides': 'ChimeraX-Nucleotides'
    , 'chimerax.open_command': 'ChimeraX-OpenCommand'
    , 'chimerax.pdb': 'ChimeraX-PDB'
    , 'chimerax.pdb_bio': 'ChimeraX-PDBBio'
    , 'chimerax.pdb_images': 'ChimeraX-PDBImages'
    , 'chimerax.pdb_lib': 'ChimeraX-PDBLibrary'
    , 'chimerax.pdb_matrices': 'ChimeraX-PDBMatrices'
    , 'chimerax.phenix': 'ChimeraX-Phenix'
    , 'chimerax.pick_blobs': 'ChimeraX-PickBlobs'
    , 'chimerax.positions': 'ChimeraX-Positions'
    , 'chimerax.preset_mgr': 'ChimeraX-PresetMgr'
    , 'chimerax.pubchem': 'ChimeraX-PubChem'
    , 'chimerax.read_pbonds': 'ChimeraX-ReadPbonds'
    , 'chimerax.realsense': 'ChimeraX-RealSense'
    , 'chimerax.registration': 'ChimeraX-Registration'
    , 'chimerax.remote_control': 'ChimeraX-RemoteControl'
    , 'chimerax.residue_fit': 'ChimeraX-ResidueFit'
    , 'chimerax.rest_server': 'ChimeraX-RestServer'
    , 'chimerax.richardson_rotamer_lib': 'ChimeraX-RotamerLibsRichardson'
    , 'chimerax.rna_layout': 'ChimeraX-RNALayout'
    , 'chimerax.rotamers': 'ChimeraX-RotamerLibMgr'
    , 'chimerax.save_command': 'ChimeraX-SaveCommand'
    , 'chimerax.scheme_mgr': 'ChimeraX-SchemeMgr'
    , 'chimerax.sdf': 'ChimeraX-SDF'
    , 'chimerax.segger': 'ChimeraX-Segger'
    , 'chimerax.segment': 'ChimeraX-Segment'
    , 'chimerax.sel_inspector': 'ChimeraX-SelInspector'
    , 'chimerax.seqalign': 'ChimeraX-Alignments'
    , 'chimerax.seqview': 'ChimeraX-SeqView'
    , 'chimerax.shape': 'ChimeraX-Shape'
    , 'chimerax.shell': 'ChimeraX-Shell'
    , 'chimerax.shortcuts': 'ChimeraX-Shortcuts'
    , 'chimerax.show_attr': 'ChimeraX-ShowAttr'
    , 'chimerax.show_sequences': 'ChimeraX-ShowSequences'
    , 'chimerax.sideview': 'ChimeraX-SideView'
    , 'chimerax.signal_viewer': 'ChimeraX-SignalViewer'
    , 'chimerax.sim_matrices': 'ChimeraX-AlignmentMatrices'
    , 'chimerax.smiles': 'ChimeraX-Smiles'
    , 'chimerax.smooth_lines': 'ChimeraX-SmoothLines'
    , 'chimerax.spacenavigator': 'ChimeraX-SpaceNavigator'
    , 'chimerax.speech': 'ChimeraX-SpeechRecognition'
    , 'chimerax.std_commands': 'ChimeraX-StdCommands'
    , 'chimerax.stl': 'ChimeraX-STL'
    , 'chimerax.storm': 'ChimeraX-Storm'
    , 'chimerax.struct_measure': 'ChimeraX-StructMeasure'
    , 'chimerax.struts': 'ChimeraX-Struts'
    , 'chimerax.surface': 'ChimeraX-Surface'
    , 'chimerax.swap_res': 'ChimeraX-SwapRes'
    , 'chimerax.swapaa': 'ChimeraX-SwapAA'
    , 'chimerax.tape_measure': 'ChimeraX-TapeMeasure'
    , 'chimerax.test': 'ChimeraX-Test'
    , 'chimerax.toolbar': 'ChimeraX-Toolbar'
    , 'chimerax.tug': 'ChimeraX-Tug'
    , 'chimerax.ui': 'ChimeraX-UI'
    , 'chimerax.uniprot': 'ChimeraX-uniprot'
    , 'chimerax.unit_cell': 'ChimeraX-UnitCell'
    , 'chimerax.viewdockx': 'ChimeraX-ViewDockX'
    , 'chimerax.viperdb': 'ChimeraX-VIPERdb'
    , 'chimerax.vive': 'ChimeraX-Vive'
    , 'chimerax.volume_menu': 'ChimeraX-VolumeMenu'
    , 'chimerax.vtk': 'ChimeraX-VTK'
    , 'chimerax.wavefront_obj': 'ChimeraX-WavefrontOBJ'
    , 'chimerax.webcam': 'ChimeraX-WebCam'
    , 'chimerax.webservices': 'ChimeraX-WebServices'
    , 'chimerax.zone': 'ChimeraX-Zone'
    , 'cxwebservices': 'ChimeraX-WebServices'
}

def main(bundle_toml: dict, write: bool = True) -> Optional[str]:
    project_info = bundle_toml['project']
    authors = project_info['authors']
    bundle_name = package_toolshed_name_map[project_info['name']]
    chimerax_info = bundle_toml['tool']['chimerax']
    b_info = '<BundleInfo name="%s" version="%s" package="%s" ' % (bundle_name, project_info['version'], project_info['name'])
    if chimerax_info.get("custom-init", False):
        b_info += 'customInit="true" '
    b_info += 'minSessionVersion="%s" maxSessionVersion="%s">\n' % (chimerax_info['min-session-version'], chimerax_info['max-session-version'])
    # Take only the first author name and email for now
    b_info += '\t<Author>%s</Author>\n' % authors[0]['name']
    b_info += '\t<Email>%s</Email>\n' % authors[0]['email']
    # Take only the first URL for now
    b_info += '\t<URL>%s</URL>\n' % list(project_info['urls'].values())[0]
    b_info += '\t<Synopsis>%s</Synopsis>\n' % project_info['synopsis']
    b_info += '\t<Description>\n%s\n\t</Description>\n' % project_info['description'].rstrip()
    b_info += '\t<Categories>\n'
    for category in chimerax_info['categories']:
        b_info += '\t\t<Category name="%s"/>\n' % category
    b_info += '\t</Categories>\n'
    b_info += '\t<Dependencies>\n'
    for dep in project_info['dependencies']:
        pkg = Requirement(dep)
        expected_pkg_name = package_toolshed_name_map[pkg.name]
        b_info += '\t\t<Dependency name="%s" version="%s"/>\n' % (expected_pkg_name, pkg.specifier)
    b_info += '\t</Dependencies>\n'
    b_info += '\t<Classifiers>\n'
    for classifier in project_info['classifiers']:
        b_info += '\t\t<PythonClassifier>%s</PythonClassifier>\n' % classifier
    for classifier in chimerax_info['chimerax-classifiers']:
        b_info += '\t\t<ChimeraXClassifier>%s</ChimeraXClassifier>\n' % classifier
    b_info += '\t</Classifiers>\n'
    b_info += '</BundleInfo>\n'
    if write:
        with open('bundle_info.xml', 'w') as bundle_info:
            bundle_info.write(b_info)
    else:
        return b_info

def toml_to_dict(toml_file):
    info = None
    try:
        with open(toml_file) as f:
            info = toml.loads(f.read())
    except toml.TomlDecodeError as e:
        print(str(e))
        sys.exit(1)
    return info

if __name__ == "__main__":
    if len(sys.argv) < 2:
        ... # print help and exit
    main(toml_to_dict(sys.argv[1]))
