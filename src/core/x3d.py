# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
x3d: support X3D output
=======================


"""
import enum


@enum.unique
class Components(enum.Enum):
    # X3D 3.0 components
    Core = 0
    DIS = 1  # Distributed interactive simulation
    EnvironmentalEffects = 2
    EnvironmentalSensor = 3
    EventUtilities = 4
    Geometry2D = 5
    Geometry3D = 6
    Geospatial = 7
    Grouping = 8
    H_Anim = 9  # Humanoid animation
    Interpolation = 10
    KeyDeviceSensor = 11
    Lighting = 12
    Navigation = 13
    Networking = 14
    NURBS = 15  # Non-rational B-Splines
    PointingDeviceSensor = 16
    Rendering = 17
    Scripting = 18
    Shape = 19
    Sound = 20
    Text = 21
    Texturing = 22
    Time = 23
    # additional 3.1 components
    CADGeometry = 24
    CubeMapTexturing = 25
    Shaders = 26
    Texturing3D = 27
    # additional 3.2 components
    Followers = 28
    Layering = 29
    Layout = 30
    RigidBodyPhysics = 31
    # additional 3.3 components
    VolumeRendering = 32
# assert(min(c.value for c in Components) == 0)
# assert(max(c.value for c in Components) == len(Components) - 1)


class X3DScene:

    def __init__(self):
        self._version = None
        self._levels = [0] * len(Components)
        self.need(Components.Core, 1)
        self._cache = {}

    def need(self, component, level):
        """say which X3D component level is needed"""
        c = component.value
        if self._levels[c] < level:
            self._levels[c] = level
            self._version = None

    def match_profile(self, profile):
        """Return if current scene fits in profile"""
        for component in Components:
            c = component.value
            if self._levels[c] > profile[c]:
                return False, component
        return True, None

    def version(self, *, minimum=(3, 0)):
        """return minimum X3D version for needed components as a string"""
        if self._version is None:
            for version, profile in FullProfiles:
                matched, mismatch = self.match_profile(profile)
                if matched:
                    break
            else:
                raise RuntimeError(mismatch.name + " component level too high")
            if version < minimum:
                version = minimum
            self._version = ".".join((str(i) for i in version))
        return self._version

    def get_profile(self, name):
        if name != 'Full':
            return Profiles[name]
        my_version = self.version()
        for version, profile in FullProfiles:
            if version == my_version:
                return profile

    def write_header(self, stream, indent, meta={}, *, namespaces={}, units={}, profile_name=None):
        # meta argument is good for author, etc.
        from html import escape
        if indent == 0:
            # only emit xml tag and doctype if not nested
            print("<?xml version='1.0' encoding='UTF-8'?>", file=stream)
            # use schema in preference to DOCTYPE
            #  "<!DOCTYPE X3D PUBLIC 'ISO//Web3D//DTD X3D 3.1//EN'"
            #  " 'http://www.web3d.org/specifications/x3d-3.1.dtd'>"

        if units:
            # The UNIT statement first appeared in version 3.3
            self.version(minimum=(3, 3))
        if profile_name is None:
            profile_name = 'Interchange'
        profile = self.get_profile(profile_name)
        tab = ' ' * indent
        print("%s<X3D version='%s' profile='%s'" % (tab, self.version(), profile_name), file=stream)
        print("%s xmlns:xsd='http://www.w3.org/2001/XMLSchema-instance'" % tab, file=stream)
        print("%s xsd:noNamespaceSchemaLocation='http://www.web3d.org/specifications/x3d-%s.xsd'" % (tab, self.version()), end='', file=stream)
        for prefix, uri in namespaces.items():
            print("\n%s xmlns:%s=\"%s\"" % (tab, escape(prefix), escape(uri)),
                  end='', file=stream)
        print('>', file=stream)
        print("%s <head>" % tab, file=stream)
        # Augment profile with component level information.
        self.write_components(stream, indent + 2, profile=profile)
        for category, (name, cf) in units.items():
            print("%s  <unit category='%s' name='%s' conversionFactor='%g'/>" % (tab, escape(category), escape(name), cf), file=stream)
        for name, content in meta.items():
            if isinstance(content, str):
                content = [content]
            for c in content:
                print("%s  <meta name=\"%s\" content=\"%s\"/>" % (tab, escape(name), escape(c)), file=stream)
        print("%s </head>" % tab, file=stream)
        print("%s <Scene>" % tab, file=stream)

    def write_footer(self, stream, indent):
        tab = ' ' * indent
        print("%s </Scene>" % tab, file=stream)
        print("%s</X3D>" % tab, file=stream)

    def write_components(self, stream, indent, profile=None):
        """write out needed components beyond what is in (Interchange) profile"""
        if profile is None:
            profile = INTERCHANGE_PROFILE
        tab = ' ' * indent
        for component in Components:
            c = component.value
            if self._levels[c] > profile[c]:
                print("%s<component name='%s' level='%s'/>"
                      % (tab, component.name, self._levels[c]),
                      file=stream)

    def def_or_use(self, key, prefix):
        name_format = '%s%d'
        if prefix not in self._cache:
            self._cache[prefix] = {key: 0}
            return 'DEF', name_format % (prefix, 0)
        prefix_cache = self._cache[prefix]
        if key not in prefix_cache:
            index = len(prefix_cache)
            prefix_cache[key] = index
            return 'DEF', name_format % (prefix, index)
        return 'USE', name_format % (prefix, prefix_cache[key])


# Appendix F, X3D Version 3.0, ISO/IEC 19775:2004
FULL_3_0_PROFILE = (
    2,  # Core
    1,  # DIS
    3,  # EnvironmentalEffects
    2,  # EnvironmentalSensor
    1,  # EventUtilities
    2,  # Geometry2D
    4,  # Geometry3D
    1,  # Geospatial
    3,  # Grouping
    1,  # H-Anim
    3,  # Interpolation
    2,  # KeyDeviceSensor
    3,  # Lighting
    2,  # Navigation
    3,  # Networking
    4,  # NURBS
    1,  # PointingDeviceSensor
    4,  # Rendering
    1,  # Scripting
    3,  # Shape
    1,  # Sound
    1,  # Text
    3,  # Texturing
    2,  # Time
)
FULL_3_0_PROFILE += (0,) * (len(Components) - len(FULL_3_0_PROFILE))

# Appendix F, X3D Version 3.1, ISO/IEC 19775:2004/Am1:2006
FULL_3_1_PROFILE = (
    2,  # Core
    1,  # DIS
    3,  # EnvironmentalEffects
    2,  # EnvironmentalSensor
    1,  # EventUtilities
    2,  # Geometry2D
    4,  # Geometry3D
    1,  # Geospatial
    3,  # Grouping
    1,  # H-Anim
    3,  # Interpolation
    2,  # KeyDeviceSensor
    3,  # Lighting
    2,  # Navigation
    3,  # Networking
    4,  # NURBS
    1,  # PointingDeviceSensor
    4,  # Rendering
    1,  # Scripting
    3,  # Shape
    1,  # Sound
    1,  # Text
    3,  # Texturing
    2,  # Time
    2,  # CADGeometry
    3,  # CubeMapTexturing
    1,  # Shaders
    2,  # Texturing3D
)
FULL_3_1_PROFILE += (0,) * (len(Components) - len(FULL_3_1_PROFILE))

# Appendix F, X3D Version 3.2, ISO/IEC 19775-1:2008, ISO/IEC 19775-2:2008
FULL_3_2_PROFILE = (
    2,  # Core
    2,  # DIS
    4,  # EnvironmentalEffects
    3,  # EnvironmentalSensor
    1,  # EventUtilities
    2,  # Geometry2D
    4,  # Geometry3D
    2,  # Geospatial
    3,  # Grouping
    1,  # H-Anim
    5,  # Interpolation
    2,  # KeyDeviceSensor
    3,  # Lighting
    3,  # Navigation
    3,  # Networking
    4,  # NURBS
    1,  # PointingDeviceSensor
    5,  # Rendering
    1,  # Scripting
    4,  # Shape
    1,  # Sound
    1,  # Text
    3,  # Texturing
    2,  # Time
    2,  # CADGeometry
    3,  # CubeMapTexturing
    1,  # Shaders
    2,  # Texturing3D
    1,  # Followers
    1,  # Layering
    2,  # Layout
    2,  # RigidBodyPhysics
)
FULL_3_2_PROFILE += (0,) * (len(Components) - len(FULL_3_2_PROFILE))

# Appendix F, X3D Version 3.3, ISO/IEC 19775-1:2008/Am1:201x, ISO/IEC 19775-1:2013
FULL_3_3_PROFILE = (
    2,  # Core
    2,  # DIS
    4,  # EnvironmentalEffects
    3,  # EnvironmentalSensor
    1,  # EventUtilities
    2,  # Geometry2D
    4,  # Geometry3D
    2,  # Geospatial
    3,  # Grouping
    1,  # H-Anim
    5,  # Interpolation
    2,  # KeyDeviceSensor
    3,  # Lighting
    3,  # Navigation
    3,  # Networking
    4,  # NURBS
    1,  # PointingDeviceSensor
    5,  # Rendering
    1,  # Scripting
    4,  # Shape
    1,  # Sound
    1,  # Text
    3,  # Texturing
    2,  # Time
    2,  # CADGeometry
    3,  # CubeMapTexturing
    1,  # Shaders
    2,  # Texturing3D
    1,  # Followers
    1,  # Layering
    2,  # Layout
    2,  # RigidBodyPhysics
    4,  # VolumeRendering
)
FULL_3_3_PROFILE += (0,) * (len(Components) - len(FULL_3_3_PROFILE))

# Appendix A, Core profile
CORE_PROFILE = (
    1,  # Core
)
CORE_PROFILE += (0,) * (len(Components) - len(CORE_PROFILE))

# Appendix B, Interchange profile
INTERCHANGE_PROFILE = (
    1,  # Core
    0,  # DIS
    1,  # EnvironmentalEffects
    0,  # EnvironmentalSensor
    0,  # EventUtilities
    0,  # Geometry2D
    2,  # Geometry3D
    0,  # Geospatial
    1,  # Grouping
    0,  # H-Anim
    2,  # Interpolation
    0,  # KeyDeviceSensor
    1,  # Lighting
    1,  # Navigation
    1,  # Networking
    0,  # NURBS
    0,  # PointingDeviceSensor
    3,  # Rendering
    0,  # Scripting
    1,  # Shape
    0,  # Sound
    0,  # Text
    2,  # Texturing
    0,  # Time
)
INTERCHANGE_PROFILE += (0,) * (len(Components) - len(INTERCHANGE_PROFILE))

# Appendix C, Interactive profile
INTERACTIVE_PROFILE = (
    1,  # Core
    0,  # DIS
    1,  # EnvironmentalEffects
    1,  # EnvironmentalSensor
    1,  # EventUtilities
    0,  # Geometry2D
    3,  # Geometry3D
    0,  # Geospatial
    2,  # Grouping
    0,  # H-Anim
    2,  # Interpolation
    1,  # KeyDeviceSensor
    2,  # Lighting
    1,  # Navigation
    2,  # Networking
    0,  # NURBS
    1,  # PointingDeviceSensor
    3,  # Rendering
    0,  # Scripting
    1,  # Shape
    0,  # Sound
    0,  # Text
    2,  # Texturing
    1,  # Time
)
INTERACTIVE_PROFILE += (0,) * (len(Components) - len(INTERACTIVE_PROFILE))

# Appendix D, MPEG-4 interactive profile
MPEG4_PROFILE = (
    1,  # Core
    0,  # DIS
    1,  # EnvironmentalEffects
    1,  # EnvironmentalSensor
    0,  # EventUtilities
    0,  # Geometry2D
    2,  # Geometry3D
    0,  # Geospatial
    2,  # Grouping
    0,  # H-Anim
    2,  # Interpolation
    0,  # KeyDeviceSensor
    2,  # Lighting
    1,  # Navigation
    2,  # Networking
    0,  # NURBS
    1,  # PointingDeviceSensor
    1,  # Rendering
    0,  # Scripting
    1,  # Shape
    0,  # Sound
    0,  # Text
    1,  # Texturing
    1,  # Time
)
MPEG4_PROFILE += (0,) * (len(Components) - len(MPEG4_PROFILE))

# Appendix E, Immersive profile
IMMERSIVE_PROFILE = (
    2,  # Core
    0,  # DIS
    2,  # EnvironmentalEffects
    2,  # EnvironmentalSensor
    1,  # EventUtilities
    1,  # Geometry2D
    4,  # Geometry3D
    0,  # Geospatial
    2,  # Grouping
    0,  # H-Anim
    2,  # Interpolation
    2,  # KeyDeviceSensor
    2,  # Lighting
    2,  # Navigation
    3,  # Networking
    0,  # NURBS
    1,  # PointingDeviceSensor
    3,  # Rendering
    1,  # Scripting
    2,  # Shape
    1,  # Sound
    1,  # Text
    3,  # Texturing
    1,  # Time
)
IMMERSIVE_PROFILE += (0,) * (len(Components) - len(IMMERSIVE_PROFILE))

# Appendix H, CADInterchange profile
CAD_PROFILE = [0] * len(Components)
CAD_PROFILE[Components.Core.value] = 1
CAD_PROFILE[Components.Networking.value] = 2
CAD_PROFILE[Components.Grouping.value] = 1
CAD_PROFILE[Components.Rendering.value] = 4
CAD_PROFILE[Components.Shape.value] = 2
CAD_PROFILE[Components.Lighting.value] = 1
CAD_PROFILE[Components.Texturing.value] = 2
CAD_PROFILE[Components.Navigation.value] = 2
CAD_PROFILE[Components.Shaders.value] = 1
CAD_PROFILE[Components.CADGeometry.value] = 2
CAD_PROFILE = tuple(CAD_PROFILE)

# Appendix L, MedicalInterchange
MEDICAL_PROFILE = [
    1,  # Core
    0,  # DIS
    1,  # EnvironmentalEffects
    0,  # EnvironmentalSensor
    1,  # EventUtilities
    2,  # Geometry2D
    2,  # Geometry3D
    0,  # Geospatial
    3,  # Grouping
    0,  # H-Anim
    2,  # Interpolation
    0,  # KeyDeviceSensor
    1,  # Lighting
    3,  # Navigation
    2,  # Networking
    0,  # NURBS
    0,  # PointingDeviceSensor
    5,  # Rendering
    0,  # Scripting
    3,  # Shape
    0,  # Sound
    1,  # Text
    2,  # Texturing
    1,  # Time
]
MEDICAL_PROFILE += [0] * (len(Components) - len(MEDICAL_PROFILE))
MEDICAL_PROFILE[Components.Texturing3D.value] = 2
MEDICAL_PROFILE[Components.VolumeRendering.value] = 4
MEDICAL_PROFILE = tuple(MEDICAL_PROFILE)

Profiles = {
    # version independent profiles
    'CADInterchange': CAD_PROFILE,
    'Core': CORE_PROFILE,
    # 'Full' profiles are version specific (below)
    'Immersive': IMMERSIVE_PROFILE,
    'Interactive': INTERACTIVE_PROFILE,
    'Interchange': INTERCHANGE_PROFILE,
    'MedicalInterchange': MEDICAL_PROFILE,
    'MPEG4Interactive': MPEG4_PROFILE,
}

FullProfiles = [
    ((3, 0), FULL_3_0_PROFILE),
    ((3, 1), FULL_3_1_PROFILE),
    ((3, 2), FULL_3_2_PROFILE),
    ((3, 3), FULL_3_3_PROFILE),
]
