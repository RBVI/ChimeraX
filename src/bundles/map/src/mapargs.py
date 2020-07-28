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

# -----------------------------------------------------------------------------
#
from chimerax.core.commands import Annotation, ModelsArg, next_token
from chimerax.core.errors import UserError

class MapsArg(ModelsArg):
    name = 'a density maps specifier'

    @classmethod
    def parse(cls, text, session):
        models, used, rest = super().parse(text, session)
        from .volume import Volume
        maps = [m for m in models if isinstance(m, Volume)]
        return maps, used, rest

class MapArg(ModelsArg):
    name = 'a density map specifier'

    @classmethod
    def parse(cls, text, session):
        models, used, rest = super().parse(text, session)
        from .volume import Volume
        maps = [m for m in models if isinstance(m, Volume)]
        if len(maps) != 1:
            raise UserError('Must specify one map, got %d' % len(maps))
        return maps[0], used, rest

class Int1or3Arg(Annotation):
    name = '1 or 3 integers'
    @staticmethod
    def parse(text, session):
        token, text, rest = next_token(text)
        try:
            s = tuple(int(f) for f in token.split(','))
        except Exception:
            raise UserError('Must specify integer or 3 comma-separated integers, got %s' % token)
        if len(s) == 1:
            s = (s[0],s[0],s[0])
        if len(s) != 3:
            raise UserError('Must specify integer or 3 comma-separated integers, got %s' % token)
        return s, text, rest

class MapStepArg(Int1or3Arg):
    name = 'map step'

class Float1or3Arg(Annotation):
    name = '1 or 3 floats'
    @staticmethod
    def parse(text, session):
        token, text, rest = next_token(text)
        try:
            s = tuple(float(f) for f in token.split(','))
        except Exception:
            raise UserError('Must specify float or 3 comma-separated floats, got %s' % token)
        if len(s) == 1:
            s = (s[0],s[0],s[0])
        if len(s) != 3:
            raise UserError('Must specify float or 3 comma-separated floats, got %s' % token)
        return s, text, rest

class Float2Arg(Annotation):
    name = '2 floats'
    @staticmethod
    def parse(text, session):
        token, text, rest = next_token(text)
        try:
            s = tuple(float(f) for f in token.split(','))
        except Exception:
            raise UserError('Must specify 2 comma-separated floats, got %s' % token)
        if len(s) != 2:
            raise UserError('Must specify 2 comma-separated floats, got %s' % token)
        return s, text, rest

class MapRegionArg(Annotation):
    name = 'map region'
    @staticmethod
    def parse(text, session):
        token, text, rest = next_token(text)
        if token == 'all' or token == 'shown':
            return token, text, rest
        try:
            s = tuple(int(f) for f in token.split(','))
        except Exception:
            raise UserError('Must specify 6 comma-separated integers, got %s' % token)
        if len(s) != 6:
            raise UserError('Must specify 6 comma-separated integers, got %s' % token)
        r = s[:3], s[3:]
        for a in (0,1,2):
            if r[0][a] > r[1][a]:
                raise UserError('Volume region axis minimum values %d,%d,%d' % r[0] +
                                ' must be less than or equal to axis maximum values %d,%d,%d' % r[1])
        return r, text, rest

class BoxArg(Annotation):
    name = 'box bounds'
    @staticmethod
    def parse(text, session):
        token, text, rest = next_token(text)
        try:
            s = tuple(float(f) for f in token.split(','))
        except Exception:
            raise UserError('Must specify 6 comma-separated floats, got %s' % token)
        if len(s) != 6:
            raise UserError('Must specify 6 comma-separated floats, got %s' % token)
        r = s[:3], s[3:]
        return r, text, rest

class ValueTypeArg(Annotation):
    name = 'numeric value type'
    @staticmethod
    def parse(text, session):
        token, text, rest = next_token(text)
        types = ('int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32',
                 'float32', 'float64')
        if token in types:
            import numpy
            vt = getattr(numpy, token)
        else:
            raise UserError('Unknown data value type "%s", use %s'
                                % (token, ', '.join(types.keys())))
        return vt, text, rest

class IntRangeArg(Annotation):
    name = 'integer range'
    @staticmethod
    def parse(text, session, default_step = 1):
        token, text, rest = next_token(text)
        try:
            s = tuple(int(f) for f in token.split(','))
        except Exception:
            s = ()
        n = len(s)
        if n < 2 or n > 3:
            raise UserError('Range argument must be 2 or 3 comma-separateed integer values, got %s' % text)
        i0,i1 = s[:2]
        step = s[2] if n >= 3 else default_step
        return (i0,i1,step), text, rest
