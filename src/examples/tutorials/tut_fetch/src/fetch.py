# vim: set expandtab shiftwidth=4 softtabstop=4:

_URL = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        "?db=homologene"
        "&id=%s"
        "&rettype=fasta"
        "&retmode=text")


def fetch_homologene(session, ident, ignore_cache=True, **kw):
    """Fetch and display sequence alignment for 'ident' from HomoloGene.

    Use Python library to download the FASTA file and use ChimeraX
    alignment tools for display.
    """
    # First fetch the file using ChimeraX core function
    url = _URL % ident
    session.logger.status("Fetching HomoloGene %s" % ident)
    save_name = "%s.fa" % ident
    from chimerax.core.fetch import fetch_file
    filename = fetch_file(session, url, "HomoloGene %s" % ident, save_name,
                "HomoloGene", ignore_cache=ignore_cache, uncompress=True)

    session.logger.status("Opening HomoloGene %s" % ident)
    models, status = session.open_command.open_data(filename, alignment=False, name=ident)
    return models, status
