# -----------------------------------------------------------------------------
#
from ..commands import Annotation, next_token
from ..errors import UserError

class MapsArg(Annotation):
    name = 'density maps'
    @staticmethod
    def parse(text, session):
        from ..commands import AtomSpecArg
        value, used, rest = AtomSpecArg.parse(text, session)
        models = value.evaluate(session).models
        from .volume import Volume
        maps = [m for m in models if isinstance(m, Volume)]
        return maps, used, rest

class MapArg(Annotation):
    name = 'density map'
    @staticmethod
    def parse(text, session):
        from ..commands import AtomSpecArg
        value, used, rest = AtomSpecArg.parse(text, session)
        models = value.evaluate(session).models
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
        except:
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
        except:
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
        except:
            raise UserError('Must specify 2 comma-separated floats, got %s' % token)
        if len(s) != 2:
            raise UserError('Must specify 2 comma-separated floats, got %s' % token)
        return s, text, rest

class MapRegionArg(Annotation):
    name = 'map region'
    @staticmethod
    def parse(text, session):
        token, text, rest = next_token(text)
        try:
            s = tuple(int(f) for f in token.split(','))
        except:
            raise UserError('Must specify 6 comma-separated integers, got %s' % token)
        if len(s) != 6:
            raise UserError('Must specify 6 comma-separated integers, got %s' % token)
        r = s[:3], s[3:]
        return r, text, rest

class BoxArg(Annotation):
    name = 'box bounds'
    @staticmethod
    def parse(text, session):
        token, text, rest = next_token(text)
        try:
            s = tuple(float(f) for f in token.split(','))
        except:
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
    default_step = 1
    @staticmethod
    def parse(text, session):
        token, text, rest = next_token(text)
        try:
            s = tuple(int(f) for f in token.split(','))
        except:
            s = ()
        n = len(s)
        if n < 2 or n > 3:
            raise UserError('Range argument must be 2 or 3 comma-separateed integer values, got %s' % text)
        i0,i1 = s[:2]
        step = s[2] if n >= 3 else default_step
        return (i0,i1,step), text, rest
