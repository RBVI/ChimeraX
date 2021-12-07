# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2021 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# Python/All
import re

# Python/Specific
from typing import Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from itertools import filterfalse

# ChimeraX/Core
from chimerax.core.commands import run

# ChimeraX/Bundles
from chimerax.atomic import AtomicStructure

# Local Imports
from . import dbparsers
from .pdbinfo import fetch_pdb_info

experimental_evidence = {
    1: "Protein Level",
    2: "Transcript Level",
    3: "Inferred from Homology",
    4: "Predicted",
    5: "Uncertain"
}

@dataclass
class Database(ABC):
    """Base class for defining blast protein databases; used to model the
    results of blast queries."""
    parser_factory: Callable[[dbparsers.Parser], object]
    parser: dbparsers.Parser = field(init=False)
    fetchable_col: str = ""
    name: str = ""
    default_cols: tuple = ("name", "e-value", "score", "description")
    # In BlastProteinWorker._process_results each hit's dict is created
    # and assigned an ID number, but we don't want to display it. It's
    # also used in BlastProteinResults._show_mav to retrieve selections.
    excluded_cols: tuple = ("id", "url")

    @abstractmethod
    def load_model(chimerax_session, match_code, ref_atomspec):
        pass

    @staticmethod
    def display_model(chimerax_session, ref_atomspec, model, chain_id):
        spec = model.atomspec
        # Rudimentary mechanism to detect NMR ensemble?
        dot_index = spec.find('.')
        if dot_index > 0: # -1 is not falsy
            spec = spec[0:dot_index]
        if chain_id:
            spec += '/' + chain_id
        if ref_atomspec:
            run(chimerax_session, "matchmaker %s to %s" %
                (spec, ref_atomspec))
        else:
            run(chimerax_session, "select add %s" % spec)

    def parse(self, query, sequence, results):
        self.parser = self.parser_factory(query, sequence, results)


@dataclass
class NCBIDB(Database):
    name: str = ""
    parser_factory: object = dbparsers.PDBParser
    fetchable_col: str = "name"
    NCBI_ID_URL: str = "https://ncbi.nlm.nih.gov/protein/%s"
    default_cols: tuple = ("hit_#", "name", "e-value", "score", "title", "resolution", "ligand_symbols")

    @staticmethod
    def load_model(chimerax_session, match_code, ref_atomspec):
        """
        url: Instance of Qt.QtCore.QUrl
        """
        # If there are two underscores only split on the first
        parts = match_code.split('_', 1)
        if len(parts) == 1 or len(parts[0]) != 4:
            chimerax_session.logger.warning("Cannot open sequence-only hit \"%s\" in model viewer" % match_code)
            return None, None
        try:
            pdb_id, chain_id = parts
        except:
            pdb_id, chain_id = parts[0], None
        models = run(chimerax_session, "open pdb:%s" % pdb_id)[0]
        if isinstance(models, AtomicStructure):
            models = [models]
        return models, chain_id

    @staticmethod 
    def format_desc(desc):
        title = species = ""
        try:
            species_range = slice(desc.rindex('['),desc.rindex(']'))
            title = desc[:species_range.start]
            species = desc[species_range.start+1:species_range.stop]
        except ValueError:
            # There is no species information in this description field
            title = desc
            species = ""
        finally:
            return title, species

    @staticmethod
    def format_formulas(formulas):
        temp_formulas = formulas.split(',')
        postprocessed_formulas = []
        for formula in temp_formulas:
            postprocessed_formulas.append(formula.replace(' ',''))
            # TODO: Activate when Trac#5407 is complete
            # Wrap the numbers in the formulas in <sub></sub> HTML tags
            #postprocessed_formulas.append(re.sub(r"([1-9]+)", lambda x: "<sub>{}</sub>".format(x.group(0)), formula).rep    lace(' ', ''))
        return ", ".join(postprocessed_formulas)
    @staticmethod
    def add_info(session, matches, sequences):
        chain_ids = list(matches.keys())
        data = fetch_pdb_info(session, chain_ids)
        for chain_id, hit in matches.items():
            for k, v in data[chain_id].items():
                if isinstance(v, list):
                    v = ", ".join([str(s) for s in v])
                hit[k] = v
            hit["title"], hit["species"] = NCBIDB.format_desc(hit["description"])
            ligand_formulas = hit.get("ligand_formulas", None)
            if ligand_formulas:
                hit["ligand_formulas"] = NCBIDB.format_formulas(ligand_formulas)
            del hit["description"]
        for hit in sequences.values():
            hit["url"] = NCBIDB.NCBI_ID_URL % hit["name"]
            hit["title"], hit["species"] = NCBIDB.format_desc(hit["description"])
            ligand_formulas = hit.get("ligand_formulas", None)
            if ligand_formulas:
                hit["ligand_formulas"] = NCBIDB.format_formulas(ligand_formulas)
            del hit["description"]


@dataclass
class PDB(NCBIDB):
    name: str = "pdb"
    pretty_name: str = "Protein Data Bank"


@dataclass
class NRDB(NCBIDB):
    name: str = "nrdb"
    pretty_name: str = "NRDB"

@dataclass
class AlphaFoldDB(Database):
    name: str = "alphafold"
    pretty_name: str = "AlphaFold Database"
    # The title of the data column that can be used to fetch the model
    fetchable_col: str = "name"
    parser_factory: object = dbparsers.AlphaFoldParser
    AlphaFold_URL: str = "https://alphafold.ebi.ac.uk/files/AF-%s-F1-model_v1.pdb"
    default_cols: tuple = ("hit_#", "name", "e-value", "score", "title", "species")
    excluded_cols: tuple = ("id", "url", "sequence_id")

    @staticmethod
    def load_model(chimerax_session, match_code, ref_atomspec):
        cmd = "alphafold fetch %s" % match_code
        if ref_atomspec:
            cmd += ' alignTo %s' % ref_atomspec
        models, _ = run(chimerax_session, cmd)
        # Hack around the fact that we use run(...) to load the model
        return models, None

    @staticmethod
    def add_info(session, matches, sequences):
        # We do not ever expect to receive sequence only hits
        for match in matches:
            raw_desc = matches[match]["description"]
            # Splitting by = then spaces lets us cut out the X=VAL attributes
            # and the longform Uniprot ID,
            hit_title = ' '.join(raw_desc.split('=')[0].split(' ')[1:-1])
            matches[match]["title"] = hit_title
            matches[match]["species"] = AlphaFoldDB._get_attr(raw_desc, 'OS')
            matches[match]["taxonomic_id"] = AlphaFoldDB._get_attr(raw_desc, 'OX')
            matches[match]["gene"] = AlphaFoldDB._get_attr(raw_desc, 'GN')
            protein_existence = AlphaFoldDB._get_attr(raw_desc, 'PE')
            matches[match]["protein_existence"] = experimental_evidence[int(protein_existence)]
            matches[match]["sequence_version"] = AlphaFoldDB._get_attr(raw_desc, 'SV')
            # At this point all useful information has been extracted from the description
            # column and formatted elsewhere.
            del matches[match]["description"]

    @staticmethod
    def _get_attr(raw_desc: str, attr: str):
        """AlphaFold's BLAST output is polluted with lots of metadata in the
        form XY=Z, in the order OS OX GN PE SV, some of which may be missing.
        This is some ugly string hacking to return the requested attr if it
        exists.

        Parameters:
            raw_desc: The description line for the hit
            attr: One of 'OS', 'OX', 'GN', 'PE', 'SV'
        """
        try:
            attr_loc = raw_desc.index("".join([attr, '=']))
        except:
            # No such attr
            return ""
        else:
            if attr_loc+2 == raw_desc.rindex('='):
                # We are at the last attribute
                try:
                    return raw_desc[attr_loc+3:]
                except:
                    # There's not even anything noted
                    return ""
            next_attr_start = raw_desc[attr_loc+3:].index('=')
            return raw_desc[attr_loc+3:][:next_attr_start-3]

AvailableDBsDict = {
    'pdb': PDB,
    'nr': NRDB,
    'alphafold': AlphaFoldDB,
}
AvailableDBs = list(AvailableDBsDict.keys())
AvailableMatrices = ["BLOSUM45", "BLOSUM50", "BLOSUM62", "BLOSUM80", "BLOSUM90", "PAM30", "PAM70", "PAM250", "IDENTITY"]

def get_database(db: str) -> Database:
    """Instantiate and return a database instance.

    Parameters:
        db: A supported database e.g 'alphafold', 'nr', 'pdb'
    """
    return AvailableDBsDict[db]()
