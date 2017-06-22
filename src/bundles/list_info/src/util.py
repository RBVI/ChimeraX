# vim: set expandtab shiftwidth=4 softtabstop=4:

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

# ==============================================================================
# Code for formatting listinfo responses

def attr_string(obj, attr):
    # Get attribute as a string
    a = getattr(obj, attr)
    if isinstance(a, str):
        # String, no action needed
        s = a
    elif isinstance(a, bytes) or isinstance(a, bytearray):
        # Binary, decode into string
        s = a.decode("utf-8")
    else:
        try:
            # Sequence, make into comma-separated string
            s = ','.join(str(v) for v in a)
        except TypeError:
            # Something else, convert to string
            s = str(a)
    # Convert into double-quote string if necessary
    l = ['"']
    need_quotes = False
    for c in s:
        if c in '"\\':
            l.append('\\')
        elif c in ' \t':
            need_quotes = True
        l.append(c)
    if need_quotes:
        l.append('"')
        return ''.join(l)
    else:
        return s

def spec(o):
    try:
        return o.atomspec()
    except AttributeError:
        try:
            return '#' + o.id_string()
        except AttributeError:
            return ""

def report_models(logger, models, attr):
    for m in models:
        try:
            value = attr_string(m, attr)
        except AttributeError:
            pass
        logger.info("model id %s type %s %s %s" % (spec(m), type(m).__name__,
                                                   attr, value))

def report_chains(logger, chains, attr):
    for c in chains:
        try:
            value = attr_string(c, attr)
        except AttributeError:
            pass
        logger.info("chain id %s %s %s" % (spec(c), attr, value))

def report_polymers(logger, polymers):
    for p in polymers:
        if len(p) < 2:
            continue
        logger.info("physical chain %s %s" % (spec(p[0]), spec(p[-1])))

def report_residues(logger, residues, attr):
    for r in residues:
        try:
            value = attr_string(r, attr)
        except AttributeError:
            continue
        info = "residue id %s %s %s" % (spec(r), attr, value)
        try:
            index = r.chain.residues.index(r)
        except (AttributeError, ValueError):
            pass
        else:
            info += " index %s" % index
        logger.info(info)

def report_atoms(logger, atoms, attr):
    for a in atoms:
        try:
            value = attr_string(a, attr)
        except AttributeError:
            pass
        else:
            logger.info("atom id %s %s %s" % (spec(a), attr, value))

def report_resattr(logger, attr):
    logger.info("resattr %s" % attr)

def report_distmat(logger, atoms, distmat):
    num_atoms = len(atoms)
    msgs = []
    for i in range(num_atoms):
        for j in range(i+1,num_atoms):
            # distmat is a scipy condensed distance matrix
            # Index calculation from answer by HongboZhu in
            # http://stackoverflow.com/questions/13079563/how-does-condensed-distance-matrix-work-pdist
            dmi = num_atoms*i - i*(i+1)//2 + j - 1 - i
            msgs.append("distmat %s %s %s" % (spec(atoms[i]), spec(atoms[j]),
                                              distmat[dmi]))
    logger.info('\n'.join(msgs))

def model_info(m):
    disp = 'shown' if m.display else 'hidden'
    line = '#%s, %s, %s' % (m.id_string(), m.name, disp)
    if m.triangles is not None:
        line += ', %d triangles' % len(m.triangles)
    npos = len(m.positions)
    if npos > 1:
        line += ', %d instances' % npos
    spos = m.selected_positions
    if spos is not None and spos.sum() > 0:
        line += ', %d selected instances' % spos.sum()
    return line

def bounds_description(bounds):
    if bounds is None:
        bdesc = 'no bounding box'
    else:
        bdesc = ('bounds %.3g,%.3g,%.3g to ' % tuple(bounds.xyz_min) +
                 '%.3g,%.3g,%.3g' % tuple(bounds.xyz_max))
    return bdesc

def structure_info(m):
    from chimerax.core.atomic import Structure
    if not isinstance(m, Structure):
        return ''

    line = ('\n%d atoms, %d bonds, %d residues, %d chains (%s)'
            % (m.num_atoms, m.num_bonds, m.num_residues, m.num_chains,
               ','.join(m.residues.unique_chain_ids)))
    ncs = m.num_coordsets
    if ncs > 1:
        line += ', %d coordsets' % ncs
    pmap = m.pbg_map
    if pmap:
        line += '\n' + ', '.join('%d %s' % (pbg.num_pseudobonds, name)
                                 for name, pbg in pmap.items())
    return line

def pseudobond_group_info(m):
    from chimerax.core.atomic import PseudobondGroup
    if isinstance(m, PseudobondGroup):
        line = ', %d pseudobonds' % m.num_pseudobonds
    else:
        line = ''
    return line

def volume_info(m):

    from chimerax.core.map import Volume
    if not isinstance(m, Volume):
        return ''

    size = 'size %d,%d,%d' % tuple(m.data.size)
    s0,s1,s2 = m.region[2]
    step = ('step %d' % s0) if s1 == s0 and s2 == s0 else 'step %d,%d,%d' % (s0,s1,s2)
    sx,sy,sz = m.data.step
    vsize = ('voxel size %.5g' % sx) if sx == sy and sy == sz else ('voxel size %.5g,%.5g,%.5g' % (sx,sy,sz))
    if m.representation == 'surface':
        level = 'level ' + ', '.join(('%.4g' % l for l in m.surface_levels))
    else:
        level = 'level/intensity ' + ', '.join(('%.4g (%.2f)' % tuple(l) for l in m.solid_levels))
    line = ' %s, %s, %s, %s' % (size, step, vsize, level)
    ms = m.matrix_value_statistics()
    line += ', value range %.5g - %.5g' % (ms.minimum, ms.maximum)
    line += ', value type %s' % str(m.data.value_type)
    sym = m.data.symmetries
    line += ', %d symmetry operators' % (len(sym) if sym else 0)
    return line

# ==============================================================================
# Code for sending REST request and waiting for response in a separate thread

from chimerax.core.tasks import Task
from contextlib import contextmanager

@contextmanager
def closing(thing):
    try:
        yield thing
    finally:
        thing.close()


class RESTTransaction(Task):

    def reset_state(self, session):
        pass

    def run(self, url, msg):
        from urllib.parse import urlencode
        from urllib.request import urlopen, URLError
        full_url = "%s?%s" % (url, urlencode([("chimerax_notification", msg)]))
        try:
            with closing(urlopen(full_url, timeout=30)) as f:
                # Discard response since we cannot handle an error anyway
                f.read()
        except URLError:
            pass


class Notifier:

    SupportedTypes = ["models", "selection"]
    # A TYPE is suppored when both _create_TYPE_handler
    # and _destroy_TYPE_handler methods are defined

    def __init__(self, what, client_id, session, prefix, url):
        self.what = what
        self.client_id = client_id
        self.session = session
        self.prefix = prefix
        self.url = url
        try:
            c_func = getattr(self, "_create_%s_handler" % what)
        except AttributeError:
            from chimerax.core.errors import UserError
            raise UserError("unsupported notification type: %s" % what)
        self._handler = c_func()
        self._handler_suspended = True
        self._destroy_handler = getattr(self, "_destroy_%s_handler" % what)
        self.Create(self)

    def start(self):
        self._handler_suspended = False
        msg = "listening for %s" % self.what
        self.session.logger.info(msg)

    def stop(self):
        self._destroy_handler()
        self.Destroy(self)
        msg = "stopped listening for %s" % self.what
        self.session.logger.info(msg)

    def suspend(self):
        self._handler_suspended = True
        msg = "suspended listening for %s" % self.what
        self.session.logger.info(msg)

    def resume(self):
        self._handler_suspended = False
        msg = "resumed listening for %s" % self.what
        self.session.logger.info(msg)

    def _notify(self, msgs):
        if self._handler_suspended:
            return
        if self.url is not None:
            # Notify via REST
            RESTTransaction(self.session).run(self.url, ''.join(msgs))
        else:
            # Just regular info messages
            logger = self.session.logger
            for msg in msgs:
                logger.info(msg)

    #
    # Methods for "models" notifications
    #
    def _create_models_handler(self):
        from chimerax.core.models import ADD_MODELS, REMOVE_MODELS
        add_handler = self.session.triggers.add_handler(ADD_MODELS,
                                                        self._notify_models)
        rem_handler = self.session.triggers.add_handler(REMOVE_MODELS,
                                                        self._notify_models)
        return [add_handler, rem_handler]

    def _destroy_models_handler(self):
        self.session.triggers.remove_handler(self._handler[0])
        self.session.triggers.remove_handler(self._handler[1])

    def _notify_models(self, trigger, trigger_data):
        msgs = []
        for m in trigger_data:
            msgs.append("%smodel %s" % (self.prefix, spec(m)))
        self._notify(msgs)

    #
    # Methods for "selection" notifications
    #
    def _create_selection_handler(self):
        from chimerax.core.selection import SELECTION_CHANGED
        handler = self.session.triggers.add_handler(SELECTION_CHANGED,
                                                    self._notify_selection)
        return handler

    def _destroy_selection_handler(self):
        self.session.triggers.remove_handler(self._handler)

    def _notify_selection(self, trigger, trigger_data):
        self._notify(["%sselection changed" % self.prefix])

    #
    # Class variables and methods for tracking by client identifier
    #

    _client_map = {}

    @classmethod
    def Find(cls, what, client_id, session=None, prefix=None, url=None):
        try:
            return cls._client_map[(what, client_id)]
        except KeyError:
            if session is None:
                from chimerax.core.errors import UserError
                raise UserError("using undefined notification client id: %s" %
                                client_id)
            return cls(what, client_id, session, prefix, url)

    @classmethod
    def Create(cls, n):
        key = (n.what, n.client_id)
        if key in cls._client_map:
            from chimerax.core.errors import UserError
            raise UserError("notification client id already in use: %s" %
                            n.client_id)
        else:
            cls._client_map[key] = n

    @classmethod
    def Destroy(cls, n):
        try:
            del cls._client_map[(n.what, n.client_id)]
        except KeyError:
            from chimerax.core.errors import UserError
            raise UserError("destroying undefined notification client id: %s" %
                            n.client_id)
