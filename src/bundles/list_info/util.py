# vim: set expandtab shiftwidth=4 softtabstop=4:

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
    from chimerax.core.atomic import Atom, Residue, Chain
    if isinstance(o, Atom):
        return spec(o.residue) + '@' + o.name
    elif isinstance(o, Residue):
        residue = ':' + str(o.number) + o.insertion_code
        if o.chain_id:
            residue = '/' + o.chain_id + residue
        return spec(o.structure) + residue
    elif isinstance(o, Chain):
        return spec(o.structure) + '/' + o.chain_id
    else:
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
            pass
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
        logger.info("atom id %s %s %s" % (spec(a), attr, value))

def report_distance(logger, ai, aj, dist):
    logger.info("distmat %s %s %s" % (spec(ai), spec(aj), dist))
