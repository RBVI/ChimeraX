from typing import Type

modality_dict = {
    # Code: Full Name, Retired, Has Image Data
    'AR': ('Autorefraction', False, True),
    'AS': ('Angioscopy', True, True),
    'ASMT': ('Content Assessment Results', False, True),
    'AU': ('Audio ECG', False, True),
    'BDUS': ('Bone Densitometry (Ultrasound)', False, True),
    'BI': ('Biomagnetic Imaging', False, True),
    'BMD': ('Bone Densitometry (X-Ray)', False, True),
    'CD': ('Color Flow Doppler', True, True),
    'CF': ('Cinefluorography', True, True),
    'CP': ('Culposcopy', True, True),
    'CR': ('Computed Radiography', False, True),
    'CS': ('Cytoscopy', True, True),
    'CT': ('Computed Tomography', False, True),
    'CTPROTOCOL': ('CT Protocol Performed', False, True),
    'DD': ('Duplex Doppler', True, True),
    'DF': ('Digital fluoroscopy', True, True),
    'DG': ('Diaphanography', False, True),
    'DM': ('Digital Microscopy', True, True),
    'DOC': ('Document', False, False),
    'DS': ('Digital Subtraction Angiography', True, True),
    'DX': ('Digital Radiography', False, True),
    'EC': ('Echocardiography', True, True),
    'ECG': ('Electrocardiography', False, True),
    'EPS': ('Cardiac Electrophysiology', False, True),
    'ES': ('Endoscopy', False, True),
    'FA': ('Fluorescein angiography', True, True),
    'FID': ('Fiducials', False, False),
    'FS': ('Fundoscopy', True, True),
    'GM': ('General Microscopy', False, True),
    'HC': ('Hard Copy', False, True),
    'HD': ('Hemodynamic Waveform', False, True),
    'IO': ('Intraoral Radiography', False, True),
    'IOL': ('Intraocular Lens Data', False, True),
    'IVOCT': ('Intravascular Optical Coherence Tomography', False, True),
    'IVUS': ('Intravascular Ultrasound', False, True),
    'KER': ('Keratometry', False, True),
    'KO': ('Key Object Selection', False, False),
    'LEN': ('Lensometry', False, True),
    'LP': ('Laparoscopy', True, True),
    'LS': ('Laser Surface Scan', False, True),
    'MA': ('Magnetic resonance angiography', True, True),
    'MG': ('Mammography', False, True),
    'MR': ('Magnetic Resonance', False, True),
    'M3D': ('Model for 3D Manufacturing', False, True),
    'MS': ('Magnetic resonance spectroscopy', True, True),
    'NM': ('Nuclear Medicine', False, True),
    'OAM': ('Ophthalmic Axial Measurements', False, True),
    'OCT': ('Optical Coherence Tomography (non-Ophthalmic)', False, True),
    'OP': ('Ophthalmic Photography', False, True),
    'OPM': ('Ophthalmic Mapping', False, True),
    'OPR': ('Ophthalmic Refraction', True, True),
    'OPT': ('Ophthalmic Tomography', False, True),
    'OPTSBV': ('Ophthalmic Tomography B-Scan Volume Analysis', False, True),
    'OPTENF': ('Ophthalmic Tomography En Face Image', False, True),
    'OPV': ('Ophthalmic Visual Field', False, True),
    'OSS': ('Optical Surface Scan', False, True),
    'OT': ('Other', False, True),
    'PLAN': ('Plan', False, False),
    'PR': ('Presentation State', False, False),
    'PT': ('Positron Emission Tomography', False, True),
    'PX': ('Panoramic X-Ray', False, True),
    'REG': ('Registration', False, False),
    'RESP': ('Respiratory Waveform', False, True),
    'RF': ('Radio Fluoroscopy', False, True),
    'RG': ('Radiographic Imaging', False, True),
    'RTDOSE': ('Radiotherapy Dose', False, True),
    'RTIMAGE': ('Radiotherapy Image', False, True),
    'RTINTENT': ('Radiotherapy Intent', False, False),
    'RTPLAN': ('Radiotherapy Plan', False, True),
    'RTRAD': ('Radiotherapy Radiation', False, True),
    'RTRECORD': ('Radiotherapy Record', False, True),
    'RTSEGANN': ('Radiotherapy Segment Annotation', False, True),
    'RTSTRUCT': ('Radiotherapy Structure Set', False, True),
    'RWV': ('Real World Value Map', False, False),
    'SEG': ('Segmentation', False, True),
    'SM': ('Slide Microscopy', False, True),
    'SMR': ('Stereometric Relationship', False, True),
    'SR': ('Structured Report', False, False),
    'SRF': ('Subjective Refraction', False, True),
    'ST': ('Single Photon Emission Computed Tomography (SPECT)', True, True),
    'STAIN': ('Automated Slide Stainer', False, True),
    'TEXTUREMAP': ('Texture Mapping', False, True),
    'TG': ('Thermography', False, True),
    'US': ('Ultrasound', False, True),
    'VA': ('Visual Acuity', False, True),
    'VF': ('Videofluorography', True, True),
    'XA': ('X-Ray Angiography', False, True),
    'XC': ('External Camera Photography', False, True),
}

class Modality(str):
    def __new__(cls: Type["Modality"], val: str) -> "Modality":
        return super().__new__(cls, val)

    @property
    def name(self) -> str:
        return modality_dict[self][0]

    @property
    def is_retired(self) -> bool:
        return modality_dict[self][1]

    @property
    def contains_image_data(self) -> bool:
        return modality_dict[self][2]

Autorefraction = Modality("AR")
ContentAssessmentResults = Modality("ASMT")
AudioECG = Modality("AU")
BoneDensitometryUltrasound = Modality("BDUS")
BiomagneticImaging = Modality("BI")
BoneDensitometryXray = Modality("BMD")
ComputedRadiography = Modality("CR")
ComputedTomography = Modality("CT")
CTProtocolPerformed = Modality("CTPROTOCOL")
Diaphanography = Modality("DG")
Document = Modality("DOC")
DigitalRadiography = Modality("DX")
Electrocardiography = Modality("ECG")
CardiacElectrophysiology = Modality("EPS")
Endoscopy = Modality("ES")
Fiducials = Modality("FID")
GeneralMicroscopy = Modality("GM")
HardCopy = Modality("HC")
HemodynamicWaveform = Modality("HD")
IntraoralRadiography = Modality("IO")
IntraocularLensData = Modality("IOL")
IntravascularOpticalCoherenceTomography = Modality("IVOCT")
IntravascularUltrasound = Modality("IVUS")
Keratometry = Modality("KER")
KeyObjectSelection = Modality("KO")
Lensometry = Modality("LEN")
LaserSurfaceScan = Modality("LS")
Mammography = Modality("MG")
MagneticResonance = Modality("MR")
ModelFor3DManufacturing = Modality("M3D")
NuclearMedicine = Modality("NM")
OphthamlicAxialMeasurements = Modality("OAM")
OpticalCoherenceTomographyNonOphthalmic = Modality("OCT")
OphthalmicPhotography = Modality("OP")
OphthalmicMapping = Modality("OPM")
OphthalmicTomography = Modality("OPT")
OphthalmicTomographyBScanVolumeAnalysis = Modality("OPTSBV")
OphthalmicTomographyEnFace = Modality("OPTENF")
OphthalmicVisualField = Modality("OPV")
OpticalSurfaceScan = Modality("OSS")
Other = Modality("OT")
Plan = Modality("PLAN")
PresentationState = Modality("PR")
PositronEmissionTomography = Modality("PT")
PanoramicXRay = Modality("PX")
Registration = Modality("REG")
RespiratoryWaveform = Modality("RESP")
RadioFluoroscopy = Modality("RF")
RadiographicImaging = Modality("RG")
RadiotherapyDose = Modality("RTDOSE")
RadiotherapyImage = Modality("RTIMAGE")
RadiotherapyIntent = Modality("RTINTENT")
RadiotherapyPlan = Modality("RTPLAN")
RadiotherapyRadiation = Modality("RTRAD")
RadiotherapyRecord = Modality("RTRECORD")
RadiotherapySegmentAnnotation = Modality("RTSEGANN")
RadiotherapyStructureSet = Modality("RTSTRUCT")
RealWorldValue = Modality('RWV')
Segmentation = Modality("SEG")
SlideMicroscopy = Modality("SM")
StereometricRelationship = Modality("SMR")
StructuredReport = Modality("SR")
SubjectiveRefraction = Modality("SRF")
AutomatedSlideStainer = Modality("STAIN")
TextureMap = Modality("TEXTUREMAP")
Thermography = Modality("TG")
Ultrasound = Modality("US")
VisualAcuity = Modality("VA")
XRayAngiography = Modality("XA")
ExternalCameraPhotography = Modality("XC")

# XA incorporates DS
# RF incorporates CF, DF, VF
# MR incorporates MA, MS
# US incorporates EC, CD, DD
# NM incorporates ST

# Retired
Angioscopy = Modality('AS')
ColorFlowDoppler = Modality("CD")
Cinefluorography = Modality('CF')
Culposcopy = Modality('CP')
Cytoscopy = Modality('CS')
DuplexDoppler = Modality("DD")
DigitalFluoroscopy = Modality('DF')
DigitalMicroscopy = Modality('DM')
DigitalSubtractionAngiography = Modality('DS')
Echocardiography = Modality('EC')
FluoresceinAngiography = Modality('FA')
Fundoscopy = Modality('FS')
Laparoscopy = Modality('LP')
MagneticResonanceAngiography = Modality("MA")
MagneticResonanceSpectroscopy = Modality("MS")
OphthalmicRefraction = Modality("OPR")
SinglePhotonEmissionComputedTomography = Modality("ST")
Videofluorography = Modality("VF")

# TCIA must-support modality list
# 'RWV', 'CT', 'SR', 'KO', 'OT', 'RTDOSE', 'SC', 'PR', 'FUSION', 'RTPLAN', 'SEG', 'REG', 'NM', 'US', 'CR', 'MR', 'DX', 'MG', 'PT', 'RTSTRUCT'

# TODO:
# SR -- Keep in metadata but don't render
# PR -- Same as SR