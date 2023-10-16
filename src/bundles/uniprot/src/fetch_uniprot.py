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

class InvalidAccessionError(ValueError):
    pass

def fetch_uniprot(session, ident, ignore_cache=False, *, associate=None):
    'Fetch UniProt data'

    from chimerax.core.errors import UserError, CancelOperation
    try:
        accession, entry = map_uniprot_ident(ident, return_value="both")
        if accession != entry:
            session.logger.info("UniProt identifier %s maps to entry %s" % (accession, entry))
        seq_string, full_name, features = fetch_uniprot_accession_info(session, entry,
            ignore_cache=ignore_cache)
    except InvalidAccessionError as e:
        raise UserError(str(e))
    except CancelOperation:
        session.logger.status("Fetch of %s cancelled" % ident)
        return
    from chimerax.atomic import Sequence
    seq = Sequence(name=ident)
    seq.extend(seq_string)
    seq.accession_id["UniProt"] = accession
    seq.set_features("UniProt", expand_features(features))
    session.logger.status("Opening UniProt %s" % ident)
    aln = session.alignments.new_alignment([seq], ident, auto_associate=(associate is None))
    if associate is not None:
        for chain in associate:
            aln.associate(chain, min_length=2)
    return [], "Opened UniProt %s" % ident

def map_uniprot_ident(ident, *, return_value="identifier"):
    from urllib.parse import urlencode
    params = {
        'from': 'UniProtKB_AC-ID',
        'to': 'UniProtKB',
        'ids': ident
    }
    data = urlencode(params)
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    request = Request("https://rest.uniprot.org/idmapping/run", bytes(data, 'utf-8'),
        headers={ "User-Agent": "Python chimerax-bugs@cgl.ucsf.edu" })
    from chimerax.core.errors import NonChimeraError
    try:
        response = urlopen(request)
    except HTTPError as e:
        raise NonChimeraError("Error from UniProt web server while submitting job: %s\n\n"
            "Try again later.  If you then still get the error, you could use"
            " Help->Report a Bug to report the error to the ChimeraX team."
            " They may be able to help you work around the problem." % e)
    job_page = response.read().decode('utf-8')
    if not job_page:
        raise InvalidAccessionError("Invalid UniProt entry name / accession number: %s" % ident)
    import json
    try:
        job_info = json.loads(job_page)
    except json.JSONDecodeError:
        raise InvalidAccessionError("Invalid UniProt entry name / accession number: %s" % ident)
    try:
        job_id = job_info['jobId']
    except KeyError:
        raise ValueError("Unexpected response from UniProt ID-mapping server: %s" % job_info)
    # wait for mapping job to finish...
    while True:
        try:
            response = urlopen("https://rest.uniprot.org/idmapping/status/" + job_id)
        except HTTPError as e:
            raise NonChimeraError("Error from UniProt web server while checking job status: %s\n\n"
                "Try again later.  If you then still get the error, you could use"
                " Help->Report a Bug to report the error to the ChimeraX team."
                " They may be able to help you work around the problem." % e)
        status_page = response.read().decode('utf-8')
        if not status_page:
            raise InvalidAccessionError("Invalid UniProt entry name / accession number: %s" % ident)
        try:
            info = json.loads(status_page)
        except json.JSONDecodeError:
            raise InvalidAccessionError("Invalid UniProt entry name / accession number: %s" % ident)
        if "jobStatus" in info:
            import time
            time.sleep(0.5)
        elif "failedIds" in info:
            raise InvalidAccessionError("Invalid UniProt entry name / accession number: %s" % ident)
        else:
            results = info['results'][0]
            if return_value == "both":
                return results['from'], results['to']['primaryAccession']
            elif return_value == "entry":
                return results['to']['primaryAccession']
            else:
                return results['from']

def fetch_uniprot_accession_info(session, accession, ignore_cache=False):
    session.logger.status("Fetch UniProt accession code %s..." % accession)
    from chimerax.core.fetch import fetch_file
    name = "%s.xml" % accession
    file_name = fetch_file(session, "https://www.uniprot.org/uniprot/%s.xml" % accession,
        "%s UniProt info" % accession, name, "UniProt", ignore_cache=ignore_cache)

    session.logger.status("Parsing %s" % name)
    import xml.dom.minidom
    tree = xml.dom.minidom.parse(file_name)
    get_child = lambda parent, tag_name: [cn for cn in parent.childNodes
        if getattr(cn, "tagName", None) == tag_name][0]
    try:
        uniprot = get_child(tree, "uniprot")
    except IndexError:
        raise InvalidAccessionError("Invalid UniProt accession number: %s" % accession)

    entry = get_child(uniprot, "entry")
    try:
        seq_node = get_child(entry, "sequence")
    except (KeyError, IndexError):
        raise AssertionError("No sequence for accession %s in UniProt info" % accession)

    protein = get_child(entry, "protein")
    rec_name = [cn for cn in protein.childNodes
        if getattr(cn, "tagName", None) in ("recommendedName", "submittedName")][0]
    full_name = get_child(rec_name, "fullName").firstChild.nodeValue
    features = [cn for cn in entry.childNodes if getattr(cn, "tagName", None) == "feature"]
    return "".join([c for c in seq_node.firstChild.nodeValue if not c.isspace()]), full_name, features

def expand_features(features):
    from chimerax.atomic.seq_support import feature_type_to_class
    expanded = {}
    feature_lookup = {}
    for feature in features:
        locs = [cn for cn in feature.childNodes if getattr(cn, 'tagName', None) == "location"]
        if not locs:
            continue
        ftype = feature.getAttribute("type")
        # try to coalesce features with the exact same attributes into one...
        attr_map = {}
        xml_attrs = feature.attributes
        strings = []
        origs = [cn for cn in feature.childNodes if getattr(cn, 'tagName', None) == "original"]
        if len(origs) == 1:
            variants = [cn for cn in feature.childNodes if getattr(cn, 'tagName', None) == "variation"]
            if len(variants) == 1:
                strings.append(origs[0].firstChild.nodeValue + "\N{RIGHTWARDS ARROW}" \
                    + variants[0].firstChild.nodeValue)
        attr_strings = []
        for attr in [xml_attrs.item(i) for i in range(xml_attrs.length)]:
            if attr.localName == "type":
                continue
            if attr.localName == "description":
                strings.append(attr.value.strip())
                continue
            attr_strings.append("%s=%s" % (attr.localName, attr.value))
        strings.extend(attr_strings)
        blocks = []
        for loc in locs:
            begin = end = None
            for cn in loc.childNodes:
                tn = getattr(cn, 'tagName', None)
                if tn == "position":
                    begin = end = int(cn.getAttribute("position"))
                elif tn == "begin" and cn.getAttribute("status") != "unknown":
                    begin = int(cn.getAttribute("position"))
                elif tn == "end" and cn.getAttribute("status") != "unknown":
                    end = int(cn.getAttribute("position"))
            if begin is None or end is None:
                continue
            blocks.append((begin, end))
        if 'bond' in ftype:
            old_blocks = blocks[:]
            blocks = []
            for block in old_blocks:
                blocks.extend([(block[0], block[0]), (block[1], block[1])])
        lookup_key = (ftype, tuple(strings))
        if lookup_key in feature_lookup:
            feature_lookup[lookup_key].positions.extend(blocks)
            continue
        f = feature_type_to_class(ftype)(strings, blocks)
        feature_lookup[lookup_key] = f
        expanded.setdefault(ftype, []).append(f)

    return expanded
