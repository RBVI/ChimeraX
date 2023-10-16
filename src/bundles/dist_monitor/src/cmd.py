# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from abc import ABC, abstractmethod

class SimpleMeasurable(ABC):
    '''
    Abstract base class used to determine if the instance has a scene_coord method
    usable in distance and angle measurements
    '''

    @property
    @abstractmethod
    def scene_coord(self):
        pass

from chimerax.atomic import Atom
SimpleMeasurable.register(Atom)

class ComplexMeasurable(ABC):
    '''
    Abstract base class for classes that have no simple XYZ coordinate to use in distance or angle
    measurements, but that may be able to offer such measurements themselves to certain other
    kinds of objects.

    The inheriting class must implement the distance(SimpleMeasurable or ComplexMeasurable)
    and angle(ComplexMeasurable) methods, either returning the corresponding value or NotImplemented
    for objects that it does not know how to make the measurement to.
    '''

    @abstractmethod
    def distance(self, obj, *, signed=False):
        pass

    @abstractmethod
    def angle(self, obj):
        pass

    @property
    def alignment_points(self):
        """ Returns two points for aligning the object along an axis.  Should be in the order
            (front, back) if sensible.
        """
        raise NotImplemented("%s does not implement alignment_points()" % self.__class__.__name__)

from chimerax.core.triggerset import TriggerSet
group_triggers = TriggerSet()
group_triggers.add_trigger("update")
group_triggers.add_trigger("delete")

def distance(session, objects, *, color=None, dashes=None,
        decimal_places=None, radius=None, symbol=None, signed=False, monitor=True):
    '''
    Show/report distance between two objects.
    '''
    from chimerax.core.errors import UserError, LimitationError
    from chimerax.centroids import CentroidModel
    from chimerax.geometry import distance
    complex_measurables = [m for m in objects.models if isinstance(m, ComplexMeasurable)]
    simple_measurables = [m for m in objects.models
        if isinstance(m, SimpleMeasurable) and not hasattr(m, 'atoms')]
    atoms = []
    centroids = []
    for atom in objects.atoms:
        if isinstance(atom.structure, CentroidModel):
            centroids.append(atom)
        else:
            atoms.append(atom)
    atomlike_measurables = atoms + simple_measurables + centroids
    non_atom_measurables = complex_measurables + simple_measurables + centroids
    measurables = complex_measurables + atoms + simple_measurables + centroids
    if len(measurables) == 2:
        if len(atomlike_measurables) != 2 or not monitor:
            # just report the distance -- no distance monitor
            if len(atomlike_measurables) == 2:
                dist = distance(measurables[0].scene_coord, measurables[1].scene_coord)
            else:
                dist = NotImplemented
                if isinstance(measurables[0], ComplexMeasurable):
                    dist = measurables[0].distance(measurables[1], signed=signed)
                if dist is NotImplemented and isinstance(measurables[1], ComplexMeasurable):
                    dist = measurables[1].distance(measurables[0], signed=signed)
                if dist is NotImplemented:
                    raise LimitationError("Don't know how to measure distance between %s and %s"
                        % tuple(measurables))
            session.logger.info(("Distance between %s and %s: " + session.pb_dist_monitor.distance_format)
                % (measurables[0], measurables[1], dist))
            return dist
        a1, a2 = measurables
        grp = session.pb_manager.get_group("distances", create=False)
        from .settings import settings
        if not grp:
            # create group and add to DistMonitor
            grp = session.pb_manager.get_group("distances")
            if color is not None:
                grp.color = color.uint8x4()
            else:
                grp.color = settings.color.uint8x4()
            if radius is not None:
                grp.radius = radius
            else:
                grp.radius = settings.radius
            grp.dashes = settings.dashes
            session.models.add([grp])
            session.pb_dist_monitor.add_group(grp, update_callback=_notify_updates)
        for pb in grp.pseudobonds:
            pa1, pa2 = pb.atoms
            if (pa1 == a1 and pa2 == a2) or (pa1 == a2 and pa2 == a1):
                raise UserError("Distance already exists;"
                    " modify distance properties with 'distance style'")
        pb = grp.new_pseudobond(a1, a2)

        if color is not None:
            pb.color = color.uint8x4()
        if dashes is not None:
            grp.dashes = dashes
        if radius is not None:
            pb.radius = radius
        if decimal_places is not None or symbol is not None:
            if decimal_places is not None:
                session.pb_dist_monitor.decimal_places = decimal_places
            if symbol is not None:
                session.pb_dist_monitor.show_units = symbol

        session.logger.info(("Distance between %s and %s: " + session.pb_dist_monitor.distance_format)
            % (a1, a2.string(relative_to=a1), pb.length))
        return pb.length
    elif len(non_atom_measurables) > 0 and len(atoms) > 0:
        results = {}
        for object in non_atom_measurables:
            dists = []
            min_info = max_info = None
            if isinstance(object, ComplexMeasurable):
                get_dist = lambda a, cm=object, signed=signed: cm.distance(a, signed=signed)
            else:
                get_dist = lambda a, al=object: distance(a.scene_coord, al.scene_coord)
            for a in atoms:
                dist = get_dist(a)
                if dist is NotImplemented:
                    break
                dists.append(dist)
                if min_info is None:
                    min_info = max_info = (dist, a)
                elif dist < min_info[0]:
                    min_info = (dist, a)
                elif dist > max_info[0]:
                    max_info = (dist, a)
            else:
                avg = sum(dists) / len(dists)
                results[object] = (min_info, avg, max_info)
                dist_fmt = session.pb_dist_monitor.distance_format
                session.logger.info(("Distance between %s and %d atoms: min " + dist_fmt + " (%s), average:"
                    " " + dist_fmt + ", max " + dist_fmt + " (%s)") % (object, len(atoms), min_info[0],
                    min_info[1], avg, max_info[0], max_info[1]))
                continue
            raise LimitationError("Don't know how to measure distance between %s and %s" % (object, a))
        return results
    else:
        raise UserError("Expected exactly two atoms and/or measurable objects (e.g. axes, planes),"
            " or one or more measurable objects and one or more atoms,"
            " got %d atoms and %d measurable objects" % (len(atoms), len(non_atom_measurables)))

def distance_save(session, save_file_name):
    from chimerax.io import open_output
    save_file = open_output(save_file_name, 'utf-8')
    from chimerax.atomic import Structure
    for model in session.models:
        if not isinstance(model, Structure):
            continue
        print("Model", model.id_string, "is", model.name, file=save_file)

    print("\nDistance information:", file=save_file)
    grp = session.pb_manager.get_group("distances", create=False)
    if grp:
        pbs = list(grp.pseudobonds)
        pbs.sort(key=lambda pb: pb.length)
        fmt = "%s <-> %s:  " + session.pb_dist_monitor.distance_format
        for pb in pbs:
            a1, a2 = pb.atoms
            d_string = fmt % (a1, a2.string(relative_to=a1), pb.length)
            # drop angstrom symbol...
            if not d_string[-1].isdigit():
                d_string = d_string[:-1]
            print(d_string, file=save_file)
    if save_file_name != save_file:
        # Wasn't a stream that was passed in...
        save_file.close()

def distance_style(session, pbonds, *, color=None, dashes=None,
        decimal_places=None, radius=None, symbol=None, set_defaults=False):
    '''
    Modify appearance of existing distance(s).
    '''
    grp = session.pb_manager.get_group("distances", create=False)
    if pbonds is not None:
        pbs = [pb for pb in pbonds if pb.group.name == "distances"]
    elif grp:
        pbs = grp.pseudobonds
    else:
        pbs = []
    from .settings import settings
    if color is not None:
        for pb in pbs:
            pb.color = color.uint8x4()
        if grp:
            from chimerax.label.label3d import labels_model, PseudobondLabel
            lm = labels_model(grp, create=False)
            if lm:
                lm.add_labels(pbs, PseudobondLabel, session.main_view,
                    settings={ 'color': color.uint8x4() })
        settings.color = color
        if set_defaults:
            settings.save('color')

    if dashes is not None:
        if not grp:
            grp = session.pb_manager.get_group("distances", create=True)
            session.models.add([grp])
            session.pb_dist_monitor.add_group(grp, update_callback=_notify_updates)
        grp.dashes = dashes
        settings.dashes = dashes
        if set_defaults:
            settings.save('dashes')

    if decimal_places is not None:
        session.pb_dist_monitor.decimal_places = decimal_places
        settings.decimal_places = decimal_places
        if set_defaults:
            settings.save('decimal_places')

    if radius is not None:
        for pb in pbs:
            pb.radius = radius
        settings.radius = radius
        if set_defaults:
            settings.save('radius')

    if symbol is not None:
        session.pb_dist_monitor.show_units = symbol
        settings.show_units = symbol
        if set_defaults:
            settings.save('show_units')

def xdistance(session, pbonds=None):
    pbg = session.pb_manager.get_group("distances", create=False)
    if not pbg:
        return
    dist_pbonds = pbonds.with_group(pbg) if pbonds != None else None
    if pbonds == None or len(dist_pbonds) == pbg.num_pseudobonds:
        session.models.close([pbg])
        group_triggers.activate_trigger('delete', None)
        return
    for pb in dist_pbonds:
        pbg.delete_pseudobond(pb)
    group_triggers.activate_trigger('delete', None)

def _notify_updates():
    group_triggers.activate_trigger('update', None)

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, AnnotationError, \
        Or, EmptyArg, ColorArg, NonNegativeIntArg, FloatArg, BoolArg, SaveFileNameArg, ObjectsArg
    from chimerax.atomic import PseudobondsArg
    d_desc = CmdDesc(
        required = [('objects', ObjectsArg)],
        keyword = [('color', ColorArg), ('dashes', NonNegativeIntArg), ('radius', FloatArg),
            ('decimal_places', NonNegativeIntArg), ('symbol', BoolArg), ('signed', BoolArg),
            ('monitor', BoolArg)],
        synopsis = 'show/report distance')
    register('distance', d_desc, distance, logger=logger)
    # command registration doesn't allow resuse of the sam CmdDesc, so...
    xd_desc = lambda: CmdDesc(
        required = [('pbonds', Or(PseudobondsArg,EmptyArg))],
        synopsis = 'remove distance monitors')
    register('~distance', xd_desc(), xdistance, logger=logger)
    register('distance delete', xd_desc(), xdistance, logger=logger)
    df_desc = CmdDesc(
        required = [('pbonds', Or(PseudobondsArg,EmptyArg))],
        keyword = [('color', ColorArg), ('dashes', NonNegativeIntArg), ('radius', FloatArg),
            ('decimal_places', NonNegativeIntArg), ('symbol', BoolArg), ('set_defaults', BoolArg)],
        synopsis = 'set distance display properties')
    register('distance style', df_desc, distance_style, logger=logger)
    ds_desc = CmdDesc(
        required = [('save_file_name', SaveFileNameArg)],
        synopsis = 'save distance information')
    register('distance save', ds_desc, distance_save, logger=logger)
