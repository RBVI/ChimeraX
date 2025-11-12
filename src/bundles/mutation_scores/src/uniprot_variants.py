# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

def fetch_uniprot_variants(session, uniprot_id, identifier = None,
                           chains = None, allow_mismatches = False, ignore_cache = False):
    '''
    Fetch UniProt variants for a UniProt entry specified by its name or accession code.  Data is in JSON format.
    Create a mutation scores instance. Example URL

       https://www.ebi.ac.uk/proteins/api/variation/Q9UNQ0
    '''
    from chimerax.atomic import is_uniprot_id
    if not is_uniprot_id(uniprot_id):
        from chimerax.core.errors import UserError
        raise UserError(f'Invalid UniProt id {uniprot_id}')
    if '_' in uniprot_id:
        # Convert uniprot name to accession code.
        from chimerax.uniprot import map_uniprot_ident, InvalidAccessionError
        try:
            uid = map_uniprot_ident(uniprot_id, return_value = 'entry')
        except InvalidAccessionError as e:
            from chimerax.core.errors import UserError
            raise UserError(str(e))
    else:
        uid = uniprot_id

    url_pattern = 'https://www.ebi.ac.uk/proteins/api/variation/%s'
    url = url_pattern % uid
    file_name = f'{uniprot_id}_variants.json'
    save_dir = 'UniProtVariants'
    from chimerax.core.fetch import fetch_file
    path = fetch_file(session, url, f'UniProt variants {uniprot_id}',
                          file_name, save_dir, ignore_cache = ignore_cache)

    mset, msg = open_uniprot_variant_scores(session, path, identifier = identifier,
                                            chains = chains, allow_mismatches = allow_mismatches)
    return mset, msg

def open_uniprot_variant_scores(session, path, identifier = None, chains = None, allow_mismatches = False):
    with open(path, 'r') as f:
        import json
        variant_info = json.load(f)

    mutation_scores = parse_uniprot_variants(session, variant_info)
    if len(mutation_scores) == 0:
        msg = f'No mutation scores in {path}'
        mset = None
        return mset, msg

    from os.path import basename, splitext
    mset_name = splitext(basename(path))[0] if identifier is None else identifier

    from .ms_data import mutation_scores_manager
    msm = mutation_scores_manager(session)
    mset = msm.mutation_set(mset_name)
    if mset is None:
        from .ms_data import MutationSet
        mset = MutationSet(mset_name, mutation_scores,
                           chains = chains, allow_mismatches = allow_mismatches, path = path)
        msm.add_scores(mset)
    else:
        mset.add_scores(mutation_scores)
        if chains:
            mset.set_associated_chains(chains, allow_mismatches)

    sinfo = []
    score_names = set()
    for ms in mutation_scores:
        score_names.update(ms.scores.keys())
    for score_name in score_names:
        v = mset.score_values(score_name)
        sinfo.append(f'{score_name} {v.count()} variants for {len(v.residue_numbers())} residues')
    msg = f'Fetched variant scores {", ".join(sinfo)}'

    return mset, msg

def parse_uniprot_variants(session, variant_info):
    '''Return a list of MutationScores instances.'''
    mscores = []
    from .ms_data import MutationScores
    features = variant_info['features']
    for variant in features:
        if variant['type'] != 'VARIANT':
            continue
        if variant['begin'] != variant['end']:
            continue  # More than one residue in variant
        if not variant.get('predictions'):
            continue  # No scores
        scores = {prediction['predAlgorithmNameType']:prediction['score'] for prediction in variant['predictions']}
        if scores:
            res_num = int(variant['begin'])
            from_aa = variant['wildType']	# One-letter amino acid code
            to_aa = variant['mutatedType']	# One-letter amino acid code
            mscores.append(MutationScores(res_num, from_aa, to_aa, scores))
    return mscores

'''
Exapmle UniProt Variants JSON output

{"accession":"Q9UNQ0",
"entryName":"ABCG2_HUMAN",
"proteinName":"Broad substrate specificity ATP-binding cassette transporter ABCG2",
"geneName":"ABCG2",
"organismName":"Homo sapiens",
"proteinExistence":"Evidence at protein level",
"sequence":"MSSSNVEVFIPVSQGNTNGFPATASNDLKAFTEGAVLSFHNICYRVKLKSGFLPCRKPVEKEILSNINGIMKPGLNAILGPTGGGKSSLLDVLAARKDPSGLSGDVLINGAPRPANFKCNSGYVVQDDVVMGTLTVRENLQFSAALRLATTMTNHEKNERINRVIQELGLDKVADSKVGTQFIRGVSGGERKRTSIGMELITDPSILFLDEPTTGLDSSTANAVLLLLKRMSKQGRTIIFSIHQPRYSIFKLFDSLTLLASGRLMFHGPAQEALGYFESAGYHCEAYNNPADFFLDIINGDSTAVALNREEDFKATEIIEPSKQDKPLIEKLAEIYVNSSFYKETKAELHQLSGGEKKKKITVFKEISYTTSFCHQLRWVSKRSFKNLLGNPQASIAQIIVTVVLGLVIGAIYFGLKNDSTGIQNRAGVLFFLTTNQCFSSVSAVELFVVEKKLFIHEYISGYYRVSSYFLGKLLSDLLPMRMLPSIIFTCIVYFMLGLKPKADAFFVMMFTLMMVAYSASSMALAIAAGQSVVSVATLLMTICFVFMMIFSGLLVNLTTIASWLSWLQYFSIPRYGFTALQHNEFLGQNFCPGLNATGNNPCNYATCTGEEYLVKQGIDLSPWGLWKNHVALACMIVIFLTIAYLKLLFLKKYS",
"sequenceChecksum":"12155046865665312168",
"sequenceVersion":3,
"taxid":9606,
"features":[
   {"type":"VARIANT",
    "alternativeSequence":"F",
    "begin":"2",
    "end":"2",
    "xrefs":[{"name":"cosmic curated",
	      "id":"COSV52949786",
	      "url":"https://cancer.sanger.ac.uk/cosmic/search?q=COSV52949786",
	      "alternativeUrl":"https://www.ensembl.org/homo_sapiens/Variation/Explore?v=COSV52949786"},
	      {"name":"gnomAD",
	      "id":"rs1212086865",
	      "url":"https://gnomad.broadinstitute.org/variant/rs1212086865?dataset=gnomad_r2_1"}
	      ],
    "cytogeneticBand":"4q22.1",
    "genomicLocation":["NC_000004.12:g.88139991G>A"],
    "locations":[
		 {"loc":"p.Ser2Phe",
		  "seqId":"ENST00000237612",
		  "source":"Ensembl"},
		 {"loc":"c.5C>T",
		 "seqId":"ENST00000237612",
		 "source":"Ensembl"},
		 {"loc":"p.Ser2Phe",
		 "seqId":"ENST00000650821",
		 "source":"Ensembl"},
		 {"loc":"c.5C>T",
		 "seqId":"ENST00000650821",
		 "source":"Ensembl"}
		 ],
    "codon":"TCT/TTT",
    "consequenceType":"missense",
    "wildType":"S",
    "mutatedType":"F",
    "predictions":[{"predictionValType":"possibly damaging",
		    "predictorType":"multi coding",
		    "score":0.744,
		    "predAlgorithmNameType":"PolyPhen",
		    "sources":["Ensembl"]},
		    {"predictionValType":"deleterious",
		    "predictorType":"multi coding",
		    "score":0.0,
		    "predAlgorithmNameType":"SIFT",
		    "sources":["Ensembl"]}
		    ],
    "somaticStatus":1,
    "sourceType":"large_scale_study"
    },

    {"type":"VARIANT", ...}
}
'''
