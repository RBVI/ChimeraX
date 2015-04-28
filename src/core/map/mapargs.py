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
            cli.UserError('Must specify one map, got %d' % len(maps))
        return maps[0], used, rest
