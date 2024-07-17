"""
Biomolecule data type.

A generalization of the 'Protein' data type in AlphaFold to general complexes that may include
multiple chains, RNA, DNA, ligands, etc.
"""
import collections
import dataclasses
import functools
import io
from typing import Any, Dict, List, Mapping, Optional, Tuple
from src.common import residue_constants
from Bio.PDB import MMCIFParser
from Bio.PDB import PDBParser
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.Structure import Structure
import numpy as np

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.

# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.


@dataclasses.dataclass(frozen=True)
class Biomolecule:
    # Cartesian coordinates of atoms in angstroms. The atom ordering
    # corresponds to the order in residue_constants.atom_types for proteins.
    # (coming soon: support for RNA and DNA)
    atom_positions: np.ndarray  # [num_token, num_max_token_atoms, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'. Ligands are encoded the same as 'X' for unknown residue.
    aatype: np.ndarray  # [num_token]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_token, num_max_token_atoms]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_token]

    # 0-indexed number corresponding to the chain in the protein that this residue
    # belongs to.
    chain_index: np.ndarray  # [num_token]

    # Unique integer for each distinct sequence.
    entity_id: np.ndarray  # [num_token]

    # Unique integer within chains of this sequence. e.g. if chains A, B, C share a sequence but
    # D does not, their sym_ids would be [0, 1, 2, 0].
    sym_id: np.ndarray  # [num_token]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_token, num_max_token_atoms]

    # A 2D matrix indicating the presence of a bond between two atoms, restricted to
    # just polymer-ligand and ligand-ligand bonds.
    token_bonds: np.ndarray  # [num_token, num_token]

    # Chemical ID of each amino-acid, nucleotide, or ligand residue represented
    # as a string. This is primarily used to record a ligand residue's name
    # (e.g., when exporting an mmCIF file from a Biomolecule object).
    chemid: np.ndarray  # [num_res]

    def __post_init__(self):
        if len(np.unique(self.chain_index)) > PDB_MAX_CHAINS:
            raise ValueError(
                f'Cannot build an instance with more than {PDB_MAX_CHAINS} chains '
                'because these cannot be written to PDB format.')


def _from_bio_structure(
        structure: Structure, chain_id: Optional[str] = None
) -> Biomolecule:
    """Takes a Biopython structure and creates a `Biomolecule` instance.

      WARNING: All non-standard residue types will be converted into UNK. All
        non-standard atoms will be ignored.

      Args:
        structure: Structure from the Biopython library.
        chain_id: If chain_id is specified (e.g. A), then only that chain is parsed.
          Otherwise, all chains are parsed.

      Returns:
        A new `Biomolecule` created from the structure contents.

      Raises:
        ValueError: If the number of models included in the structure is not 1.
        ValueError: If insertion code is detected at a residue.
      """
    pass


def from_pdb_string(pdb_str: str, chain_id: Optional[str] = None) -> Biomolecule:
    """Takes a PDB string and constructs a `Biomolecule` object.

    WARNING: All non-standard residue types will be converted into UNK. All
        non-standard atoms will be ignored.

    Args:
        pdb_str: The contents of the pdb file
        chain_id: If chain_id is specified (e.g. A), then only that chain is parsed.
            Otherwise, all chains are parsed.

    Returns:
        A new `Biomolecule` parsed from the pdb contents.
  """
    with io.StringIO(pdb_str) as pdb_fh:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(id='none', file=pdb_fh)
        return _from_bio_structure(structure, chain_id)


def from_mmcif_string(
        mmcif_str: str, chain_id: Optional[str] = None
) -> Biomolecule:
    """Takes a mmCIF string and constructs a `Protein` object.

    WARNING: All non-standard residue types will be converted into UNK. All
        non-standard atoms will be ignored.

    Args:
        mmcif_str: The contents of the mmCIF file
        chain_id: If chain_id is specified (e.g. A), then only that chain is parsed.
            Otherwise, all chains are parsed.

    Returns:
        A new `Protein` parsed from the mmCIF contents.
  """
    with io.StringIO(mmcif_str) as mmcif_fh:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(structure_id='none', filename=mmcif_fh)
        return _from_bio_structure(structure, chain_id)


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
    chain_end = 'TER'
    return (f'{chain_end:<6}{atom_index:>5}      {end_resname:>3} '
            f'{chain_name:>1}{residue_index:>4}')


def to_pdb(biomol: Biomolecule) -> str:
    """Converts a `Biomolecule` instance to a PDB string.

      Args:
        biomol: The biomolecule to convert to PDB.

      Returns:
        PDB string.
      """
    pass


def ideal_atom_mask(biomol: Biomolecule) -> np.ndarray:
    """Computes an ideal atom mask.

    `Protein.atom_mask` typically is defined according to the atoms that are
    reported in the PDB. This function computes a mask according to heavy atoms
    that should be present in the given sequence of amino acids.

    Args:
        biomol: `Protein` whose fields are `numpy.ndarray` objects.

    Returns:
        An ideal atom mask.
    """
    return residue_constants.STANDARD_ATOM_MASK[biomol.aatype]


def from_prediction(
        features: FeatureDict,
        result: ModelOutput,
        b_factors: Optional[np.ndarray] = None,
        remove_leading_feature_dimension: bool = True) -> Biomolecule:
    """Assembles a protein from a prediction.
    Args:
        features: Dictionary holding model inputs.
        result: Dictionary holding model outputs.
        b_factors: (Optional) B-factors to use for the protein.
        remove_leading_feature_dimension: Whether to remove the leading dimension
            of the `features` values.

    Returns:
        A protein instance.
    """
    pass


def to_mmcif(
        biomol: Biomolecule,
        file_id: str,
        model_type: str
) -> str:
    """Converts a `Biomolecule` instance to an mmCIF string.
    WARNING 1: The _entity_poly_seq is filled with unknown (UNK) residues for any
        missing residue indices in the range from min(1, min(residue_index)) to
        max(residue_index). E.g. for a protein object with positions for residues
        2 (MET), 3 (LYS), 6 (GLY), this method would set the _entity_poly_seq to:
        1 UNK
        2 MET
        3 LYS
        4 UNK
        5 UNK
        6 GLY
        This is done to preserve the residue numbering.

    WARNING 2: Converting ground truth mmCIF file to Biomolecule and then back to
        mmCIF using this method will convert all non-standard residue types to UNK.
        If you need this behaviour, you need to store more mmCIF metadata in the
        Protein object (e.g. all fields except for the _atom_site loop).

    WARNING 3: Converting ground truth mmCIF file to Biomolecule and then back to
        mmCIF using this method will not retain the original chain indices.

    WARNING 4: In case of multiple identical chains, they are assigned different
        `_atom_site.label_entity_id` values.

    Args:
        biomol: A biomolecule to convert to mmCIF string.
        file_id: The file ID (usually the PDB ID) to be used in the mmCIF.
        model_type: 'Multimer' or 'Monomer'.

    Returns:
        A valid mmCIF string.

    Raises:
        ValueError: If aminoacid types array contains entries with too many protein
        types.
    """
    pass


# Other methods that make the methods above possible.
