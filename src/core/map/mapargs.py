# -----------------------------------------------------------------------------
#
from .. import cli
class MapsArg(cli.Annotation):
    name = 'density maps'
    @staticmethod
    def parse(text, session):
        from ..atomspec import AtomSpecArg
        value, used, rest = AtomSpecArg.parse(text, session)
        models = value.evaluate(session).models
        from .volume import Volume
        maps = [m for m in models if isinstance(m, Volume)]
        return maps, used, rest

class MapArg(cli.Annotation):
    name = 'density map'
    @staticmethod
    def parse(text, session):
        from ..atomspec import AtomSpecArg
        value, used, rest = AtomSpecArg.parse(text, session)
        models = value.evaluate(session).models
        from .volume import Volume
        maps = [m for m in models if isinstance(m, Volume)]
        if len(maps) != 1:
            raise cli.UserError('Must specify one map, got %d' % len(maps))
        return maps[0], used, rest

class Int1or3Arg(cli.Annotation):
    name = '1 or 3 integers'
    @staticmethod
    def parse(text, session):
        token, text, rest = cli.next_token(text)
        try:
            s = tuple(int(f) for f in token.split(','))
        except:
            raise cli.UserError('Must specify integer or 3 comma-separated integers, got %s' % token)
        if len(s) == 1:
            s = (s[0],s[0],s[0])
        if len(s) != 3:
            raise cli.UserError('Must specify integer or 3 comma-separated integers, got %s' % token)
        return s, text, rest

class MapStepArg(Int1or3Arg):
    name = 'map step'

class Float1or3Arg(cli.Annotation):
    name = '1 or 3 floats'
    @staticmethod
    def parse(text, session):
        token, text, rest = cli.next_token(text)
        try:
            s = tuple(float(f) for f in token.split(','))
        except:
            raise cli.UserError('Must specify float or 3 comma-separated floats, got %s' % token)
        if len(s) == 1:
            s = (s[0],s[0],s[0])
        if len(s) != 3:
            raise cli.UserError('Must specify float or 3 comma-separated floats, got %s' % token)
        return s, text, rest

class Float2Arg(cli.Annotation):
    name = '2 floats'
    @staticmethod
    def parse(text, session):
        token, text, rest = cli.next_token(text)
        try:
            s = tuple(float(f) for f in token.split(','))
        except:
            raise cli.UserError('Must specify 2 comma-separated floats, got %s' % token)
        if len(s) != 2:
            raise cli.UserError('Must specify 2 comma-separated floats, got %s' % token)
        return s, text, rest

class MapRegionArg(cli.Annotation):
    name = 'map region'
    @staticmethod
    def parse(text, session):
        token, text, rest = cli.next_token(text)
        try:
            s = tuple(int(f) for f in token.split(','))
        except:
            raise cli.UserError('Must specify 6 comma-separated integers, got %s' % token)
        if len(s) != 6:
            raise cli.UserError('Must specify 6 comma-separated integers, got %s' % token)
        r = s[:3], s[3:]
        return r, text, rest

class BoxArg(cli.Annotation):
    name = 'box bounds'
    @staticmethod
    def parse(text, session):
        token, text, rest = cli.next_token(text)
        try:
            s = tuple(float(f) for f in token.split(','))
        except:
            raise cli.UserError('Must specify 6 comma-separated floats, got %s' % token)
        if len(s) != 6:
            raise cli.UserError('Must specify 6 comma-separated floats, got %s' % token)
        r = s[:3], s[3:]
        return r, text, rest

class ValueTypeArg(cli.Annotation):
    name = 'numeric value type'
    @staticmethod
    def parse(text, session):
        token, text, rest = cli.next_token(text)
        types = ('int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32',
                 'float32', 'float64')
        if token in types:
            import numpy
            vt = getattr(numpy, token)
        else:
            raise cli.UserError('Unknown data value type "%s", use %s'
                                % (token, ', '.join(types.keys())))
        return vt, text, rest
