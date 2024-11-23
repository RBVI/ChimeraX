from chimerax.core.state import State
from chimerax.core.objects import all_objects


class SceneColors(State):
    """
    Manages color data for session objects in ChimeraX.

    This class stores and restores color data for atoms, bonds, pseudobonds, ribbons, and rings. It provides methods to
    initialize color data, restore color states, handle model removal, and check if two SceneColors instances are
    interpolatable. It also supports interpolation between two color states.

    Attributes:
        session: The current session.
        atom_colors: A dictionary storing the color state of atoms.
        bond_colors: A dictionary storing the color state of bonds.
        halfbonds: A dictionary storing the halfbond drawing style state of bonds.
        pseudobond_colors: A dictionary storing the color state of pseudobonds.
        pbond_halfbonds: A dictionary storing the halfbond drawing style state of pseudobonds.
        ribbon_colors: A dictionary storing the color state of ribbons.
        ring_colors: A dictionary storing the color state of rings.
    """

    # TODO After testing consider making this class more concise.

    version = 0

    def __init__(self, session, color_data=None):
        """
        Initialize a SceneColors object. If color data is provided from a snapshot, it initializes the object with
        that data. Otherwise, it initializes with current session values.

        Args:
            session: The current session.
            color_data (dict, optional): A dictionary containing color data to initialize the object from snapshot.
        """
        self.session = session

        if color_data:
            self.atom_colors = color_data['atom_colors']
            self.bond_colors = color_data['bond_colors']
            self.halfbonds = color_data['halfbonds']
            self.pseudobond_colors = color_data['pseudobond_colors']
            self.pbond_halfbonds = color_data['pbond_halfbonds']
            self.ribbon_colors = color_data['ribbon_colors']
            self.ring_colors = color_data['ring_colors']
        else:
            # Atom colors
            self.atom_colors = {}

            # Bond colors
            self.bond_colors = {}
            self.halfbonds = {}  # Boolean ndarray indicating half bond drawing style per bond

            # Pseudobond colors
            self.pseudobond_colors = {}
            self.pbond_halfbonds = {}  # Boolean ndarray indicating half bond drawing style per pseudobond

            # Residue colors
            self.ribbon_colors = {}
            self.ring_colors = {}

            self.initialize_colors()
        return

    def initialize_colors(self):
        """
        Initialize values for the color attributes. Collections from the C++ layer have a by_structure attribute which
        maps models to their respective object pointers. This method uses this mapping to store the colors for each
        model and then restore them later.
        """

        objects = all_objects(self.session)

        # Atoms Colors
        for (model, atoms) in objects.atoms.by_structure:
            self.atom_colors[model] = atoms.colors

        # Bonds colors
        for (model, bonds) in objects.bonds.by_structure:
            self.bond_colors[model] = bonds.colors
            self.halfbonds[model] = bonds.halfbonds  # Boolean ndarray indicating half bond drawing style per bond

        # Pseudobonds colors
        for (pbond_group, pseudobonds) in objects.pseudobonds.by_group:
            self.pseudobond_colors[pbond_group] = pseudobonds.colors
            self.pbond_halfbonds[pbond_group] = pseudobonds.halfbonds

        # Residue Colors
        for (model, residues) in objects.residues.by_structure:
            self.ribbon_colors[model] = residues.ribbon_colors
            self.ring_colors[model] = residues.ring_colors

    def restore_colors(self):
        """
        Restore the color state of all session objects saved in this class.
        """

        objects = all_objects(self.session)

        # Atoms colors
        for (model, atoms) in objects.atoms.by_structure:
            if model in self.atom_colors.keys():
                atoms.colors = self.atom_colors[model]

        # Bonds colors
        for (model, bonds) in objects.bonds.by_structure:
            if model in self.bond_colors.keys():
                bonds.colors = self.bond_colors[model]
                bonds.halfbonds = self.halfbonds[model]

        # Pseudobonds colors
        for (pbond_group, pseudobonds) in objects.pseudobonds.by_group:
            if pbond_group in self.pseudobond_colors.keys():
                pseudobonds.colors = self.pseudobond_colors[pbond_group]
                pseudobonds.halfbonds = self.pbond_halfbonds[pbond_group]

        # Residue Colors
        for (model, residues) in objects.residues.by_structure:
            if model in self.ribbon_colors.keys():
                residues.ribbon_colors = self.ribbon_colors[model]
                residues.ring_colors = self.ring_colors[model]

    def models_removed(self, models: [str]):
        """
        Remove models and associated color data when models are deleted. Designed to be attached to a handler for the
        models removed trigger.

        Args:
            models (list of str): List of model identifiers to remove.
        """

        for model in models:
            if model in self.atom_colors:
                del self.atom_colors[model]

            if model in self.bond_colors:
                del self.bond_colors[model]
            if model in self.halfbonds:
                del self.halfbonds[model]

            if model in self.pseudobond_colors:
                del self.pseudobond_colors[model]
            if model in self.pbond_halfbonds:
                del self.pbond_halfbonds[model]

            if model in self.ribbon_colors:
                del self.ribbon_colors[model]
            if model in self.ring_colors:
                del self.ring_colors[model]

    def get_atom_colors(self):
        return self.atom_colors

    def get_bond_colors(self):
        return self.bond_colors

    def get_halfbonds(self):
        return self.halfbonds

    def get_pseudobond_colors(self):
        return self.pseudobond_colors

    def get_pbond_halfbonds(self):
        return self.pbond_halfbonds

    def get_ribbon_colors(self):
        return self.ribbon_colors

    def get_ring_colors(self):
        return self.ring_colors

    def take_snapshot(self, session, flags):
        return {
            'version': self.version,
            'atom_colors': self.atom_colors,
            'bond_colors': self.bond_colors,
            'halfbonds': self.halfbonds,
            'pseudobond_colors': self.pseudobond_colors,
            'pbond_halfbonds': self.pbond_halfbonds,
            'ribbon_colors': self.ribbon_colors,
            'ring_colors': self.ring_colors
        }

    @staticmethod
    def restore_snapshot(session, data):
        if data['version'] != SceneColors.version:
            raise ValueError("Cannot restore SceneColors data with version %d" % data['version'])
        return SceneColors(session, color_data=data)


class SceneVisibility(State):
    """
    Manages the visibility state of various session objects in ChimeraX.

    This class stores and restores visibility data for models, atoms, bonds, pseudobonds, ribbons, and rings. It
    provides methods to initialize visibility data, restore visibility states, handle model removal, and check if two
    SceneVisibility instances are interpolatable. It also supports interpolation between two visibility states.

    Attributes:
        session: The current session.
        model_visibility: A dictionary storing the visibility state of models.
        atom_displays: A dictionary storing the display state of atoms.
        bond_displays: A dictionary storing the display state of bonds.
        pseudobond_displays: A dictionary storing the display state of pseudobonds.
        ribbon_displays: A dictionary storing the display state of ribbons.
        ring_displays: A dictionary storing the display state of rings.
    """

    version = 0

    def __init__(self, session, *, visibility_data=None):
        """
        Initialize a SceneVisibility object. If visibility snapshot data is provided, it initializes the object with
        that data. Otherwise, it initializes with current session values.

        Args:
            session: The current session. visibility_data (dict, optional): A dictionary from take_snapshot
            containing visibility data to initialize the object.
        """
        self.session = session
        if visibility_data:
            self.model_visibility = visibility_data['model_visibility']
            self.atom_displays = visibility_data['atom_displays']
            self.bond_displays = visibility_data['bond_displays']
            self.pseudobond_displays = visibility_data['pseudobond_displays']
            self.ribbon_displays = visibility_data['ribbon_displays']
            self.ring_displays = visibility_data['ring_displays']
        else:
            self.model_visibility = {}
            self.atom_displays = {}
            self.bond_displays = {}
            self.pseudobond_displays = {}
            self.ribbon_displays = {}
            self.ring_displays = {}
            self.initialize_visibility()
        return

    def initialize_visibility(self):
        """
        Initialize visibility data for all session objects from current session.
        """
        objects = all_objects(self.session)

        for model in objects.models:
            self.model_visibility[model] = model.display
        for (structure, atom) in objects.atoms.by_structure:
            self.atom_displays[structure] = atom.displays
        for (structure, bond) in objects.bonds.by_structure:
            self.bond_displays[structure] = bond.displays
        for (pbond_group, pseudobonds) in objects.pseudobonds.by_group:
            self.pseudobond_displays[pbond_group] = pseudobonds.displays
        for (structure, residues) in objects.residues.by_structure:
            self.ribbon_displays[structure] = residues.ribbon_displays
            self.ring_displays[structure] = residues.ring_displays

        return

    def restore_visibility(self):
        """
        Restore the visibility state of all session objects saved in this class.
        """
        objects = all_objects(self.session)

        for model in objects.models:
            if model in self.model_visibility:
                model.display = self.model_visibility[model]
        for (structure, atom) in objects.atoms.by_structure:
            if structure in self.atom_displays:
                atom.displays = self.atom_displays[structure]
        for (structure, bond) in objects.bonds.by_structure:
            if structure in self.bond_displays:
                bond.displays = self.bond_displays[structure]
        for (pbond_group, pseudobonds) in objects.pseudobonds.by_group:
            if pbond_group in self.pseudobond_displays:
                pseudobonds.displays = self.pseudobond_displays[pbond_group]
        for (structure, residues) in objects.residues.by_structure:
            if structure in self.ribbon_displays:
                residues.ribbon_displays = self.ribbon_displays[structure]
            if structure in self.ring_displays:
                residues.ring_displays = self.ring_displays[structure]

    def models_removed(self, models: [str]):
        """
        Remove visibility data for specified models. Designed to be attached to a handler for the models removed
        trigger.

        Args:
            models (list of str): List of model identifiers to remove.
        """
        for model in models:
            if model in self.model_visibility:
                del self.model_visibility[model]
            if model in self.atom_displays:
                del self.atom_displays[model]
            if model in self.bond_displays:
                del self.bond_displays[model]
            if model in self.pseudobond_displays:
                del self.pseudobond_displays[model]
            if model in self.ribbon_displays:
                del self.ribbon_displays[model]
            if model in self.ring_displays:
                del self.ring_displays[model]

    def get_model_visibility(self):
        return self.model_visibility

    def get_atom_displays(self):
        return self.atom_displays

    def get_bond_displays(self):
        return self.bond_displays

    def get_pbond_displays(self):
        return self.pseudobond_displays

    def get_ribbon_displays(self):
        return self.ribbon_displays

    def get_ring_displays(self):
        return self.ring_displays

    def take_snapshot(self, session, flags):
        return {
            'version': self.version,
            'model_visibility': self.model_visibility,
            'atom_displays': self.atom_displays,
            'bond_displays': self.bond_displays,
            'pseudobond_displays': self.pseudobond_displays,
            'ribbon_displays': self.ribbon_displays,
            'ring_displays': self.ring_displays
        }

    @staticmethod
    def restore_snapshot(session, data):
        if SceneVisibility.version != data['version']:
            raise ValueError("Cannot restore SceneVisibility data with version %d" % data['version'])
        return SceneVisibility(session, visibility_data=data)
