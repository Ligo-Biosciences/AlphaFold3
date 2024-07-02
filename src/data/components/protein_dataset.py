"""Subclasses of `torch.utils.data.Dataset` and `torch.utils.data.DataLoader` that are tuned for loading proteinflow
data. """

import os
import pickle
import random
from collections import defaultdict
from copy import deepcopy
from itertools import combinations, groupby
from operator import itemgetter

import numpy as np
import torch
from p_tqdm import p_map, t_map
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from proteinflow.constants import ALPHABET, CDR_REVERSE, D3TO1, MAIN_ATOMS
from proteinflow.data import ProteinEntry


class ProteinDataset(Dataset):
    """Dataset to load proteinflow data.

    Saves the model x tensors as pickle files in `features_folder`. When `clustering_dict_path` is provided,
    at each iteration a random biounit from a cluster is sampled.

    If a complex contains multiple chains, they are concatenated. The sequence identity information is preserved in the
    `'chain_encoding_all'` object and in the `'residue_idx'` arrays the chain change is denoted by a +100 jump.

    Returns dictionaries with the following keys and values (all values are `torch` tensors):

    - `'X'`: 3D coordinates of N, C, Ca, O, `(total_L, 4, 3)`,
    - `'S'`: sequence indices (shape `(total_L)`),
    - `'mask'`: residue mask (0 where coordinates are missing, 1 otherwise; with interpolation 0s are replaced with 1s), `(total_L)`,
    - `'mask_original'`: residue mask (0 where coordinates are missing, 1 otherwise; not changed with interpolation), `(total_L)`,
    - `'residue_idx'`: residue indices (from 0 to length of sequence, +100 where chains change), `(total_L)`,
    - `'chain_encoding_all'`: chain indices, `(total_L)`,
    - `'chain_id`': a sampled chain index,
    - `'chain_dict'`: a dictionary of chain ids (keys are chain ids, e.g. `'A'`, values are the indices used in `'chain_id'` and `'chain_encoding_all'` objects)

    You can also choose to include additional features (set in the `node_features_type` parameter):

    - `'sidechain_orientation'`: a unit vector in the direction of the sidechain, `(total_L, 3)`,
    - `'dihedral'`: the dihedral angles, `(total_L, 2)`,
    - `'chemical'`: hydropathy, volume, charge, polarity, acceptor/donor features, `(total_L, 6)`,
    - `'secondary_structure'`: a one-hot encoding of secondary structure ([alpha-helix, beta-sheet, coil]), `(total_L, 3)`,
    - `'sidechain_coords'`: the coordinates of the sidechain atoms (see `proteinflow.sidechain_order()` for the order), `(total_L, 10, 3)`.

    If the dataset contains a `'cdr'` key (if it was generated from SAbDab files), the output files will also additionally contain a `'cdr'`
    key with a CDR tensor of length `total_L`. In the array, the CDR residues are marked with the corresponding CDR type
    (H1=1, H2=2, H3=3, L1=4, L2=5, L3=6) and the rest of the residues are marked with 0s.

    Use the `set_cdr` method to only iterate over specific CDRs.

    In order to compute additional features, use the `feature_functions` parameter. It should be a dictionary with keys
    corresponding to the feature names and values corresponding to the functions that compute the features. The functions
    should take a `proteinflow.data.ProteinEntry` instance and a list of chains and return a `numpy` array shaped as `(#residues, #features)`
    where `#residues` is the total number of residues in those chains and the features are concatenated in the order of the list:
    `func(data_entry: ProteinEntry, chains: list) -> np.ndarray`.

    If `mask_residues` is `True`, an additional `'masked_res'` key is added to the output. The value is a binary
    tensor shaped `(B, L)` where 1 denotes the part that needs to be predicted and 0 is everything else. The tensors are generated
    according to the following rulesd:
    - if the dataset is generated from SAbDab files, the sampled CDR is masked,
    - if `mask_whole_chains` is `True`, the whole chain is masked,
    - if `mask_frac` is given, the number of residues to mask is `mask_frac` times the length of the chain,
    - otherwise, the number of residues to mask is sampled uniformly from the range [`lower_limit`, `upper_limit`].

    If `force_binding_sites_frac` > 0 and `mask_whole_chains` is `False`, in the fraction of cases where a chain
    from a polymer is sampled, the center of the masked region will be forced to be in a binding site (in PDB datasets).

    """

    def __init__(
        self,
        dataset_folder,
        features_folder="./data/tmp/",
        clustering_dict_path=None,
        max_length=None,
        rewrite=False,
        use_fraction=1,
        load_to_ram=False,
        debug=False,
        interpolate="none",
        node_features_type="zeros",
        debug_file_path=None,
        entry_type="biounit",  # biounit, chain, pair
        classes_to_exclude=None,  # heteromers, homomers, single_chains
        shuffle_clusters=True,
        min_cdr_length=None,
        feature_functions=None,
        classes_dict_path=None,
        cut_edges=False,
        mask_residues=True,
        lower_limit=15,
        upper_limit=100,
        mask_frac=None,
        mask_whole_chains=False,
        mask_sequential=False,
        force_binding_sites_frac=0.15,
        mask_all_cdrs=False,
        load_ligands=False,
        pyg_graph=False,
        patch_around_mask=False,
        initial_patch_size=128,
        antigen_patch_size=128,
        require_antigen=False,
        require_light_chain=False,
        require_no_light_chain=False,
        require_heavy_chain=False,
    ):
        """Initialize the dataset.

        Parameters
        ----------
        dataset_folder : str
            the path to the folder with proteinflow format x files (assumes that files are named {biounit_id}.pickle)
        features_folder : str, default "./data/tmp/"
            the path to the folder where the ProteinMPNN features will be saved
        clustering_dict_path : str, optional
            path to the pickled clustering dictionary (keys are cluster ids, values are (biounit id, chain id) tuples)
        max_length : int, optional
            entries with total length of chains larger than `max_length` will be disregarded
        rewrite : bool, default False
            if `False`, existing feature files are not overwritten
        use_fraction : float, default 1
            the fraction of the clusters to use (first N in alphabetic order)
        load_to_ram : bool, default False
            if `True`, the data will be stored in RAM (use with caution! if RAM isn'timesteps big enough the machine might crash)
        debug : bool, default False
            only process 1000 files
        interpolate : {"none", "only_middle", "all"}
            `"none"` for no interpolation, `"only_middle"` for only linear interpolation in the middle, `"all"` for linear interpolation + ends generation
        node_features_type : {"zeros", "dihedral", "sidechain_orientation", "chemical", "secondary_structure" or combinations with "+"}
            the type of node features, e.g. `"dihedral"` or `"sidechain_orientation+chemical"`
        debug_file_path : str, optional
            if not `None`, open this single file instead of loading the dataset
        entry_type : {"biounit", "chain", "pair"}
            the type of entries to generate (`"biounit"` for biounit-level complexes, `"chain"` for chain-level, `"pair"`
            for chain-chain pairs (all pairs that are seen in the same biounit and have intersecting coordinate clouds))
        classes_to_exclude : list of str, optional
            a list of classes to exclude from the dataset (select from `"single_chain"`, `"heteromer"`, `"homomer"`)
        shuffle_clusters : bool, default True
            if `True`, a new representative is randomly selected for each cluster at each epoch (if `clustering_dict_path` is given)
        min_cdr_length : int, optional
            for SAbDab datasets, biounits with CDRs shorter than `min_cdr_length` will be excluded
        feature_functions : dict, optional
            a dictionary of functions to compute additional features (keys are the names of the features, values are the functions)
        classes_dict_path : str, optional
            a path to a pickled dictionary with biounit classes (single chain / heteromer / homomer)
        cut_edges : bool, default False
            if `True`, missing values at the edges of the sequence will be cut off
        mask_residues : bool, default True
            if `True`, the masked residues will be added to the output
        lower_limit : int, default 15
            the lower limit of the number of residues to mask
        upper_limit : int, default 100
            the upper limit of the number of residues to mask
        mask_frac : float, optional
            if given, the number of residues to mask is `mask_frac` times the length of the chain
        mask_whole_chains : bool, default False
            if `True`, the whole chain is masked
        mask_sequential : bool, default False
            if `True`, the masked residues will be neighbors in the sequence; otherwise a geometric
            mask is applied based on the coordinates
        force_binding_sites_frac : float, default 0.15
            if `force_binding_sites_frac` > 0 and `mask_whole_chains` is `False`, in the fraction of cases where a chain
            from a polymer is sampled, the center of the masked region will be forced to be in a binding site (in PDB datasets)
        mask_all_cdrs : bool, default False
            if `True`, all CDRs will be masked (in SAbDab datasets)
        load_ligands : bool, default False
            if `True`, the ligands will be loaded as well
        pyg_graph : bool, default False
            if `True`, the output will be a `torch_geometric.data.Data` object instead of a dictionary
        patch_around_mask : bool, default False
            if `True`, the data entries will be cut around the masked region
        initial_patch_size : int, default 128
            the size of the initial patch (used if `patch_around_mask` is `True`)
        antigen_patch_size : int, default 128
            the size of the antigen patch (used if `patch_around_mask` is `True` and the dataset is SAbDab)
        require_antigen : bool, default False
            if `True`, only entries with an antigen will be included (used if the dataset is SAbDab)
        require_light_chain : bool, default False
            if `True`, only entries with a light chain will be included (used if the dataset is SAbDab)
        require_no_light_chain : bool, default False
            if `True`, only entries without a light chain will be included (used if the dataset is SAbDab)
        require_heavy_chain : bool, default False
            if `True`, only entries with a heavy chain will be included (used if the dataset is SAbDab)

        """
        self.debug = False

        if classes_dict_path is None:
            dataset_parent = os.path.dirname(dataset_folder)
            classes_dict_path = os.path.join(
                dataset_parent, "splits_dict", "classes.pickle"
            )
            if not os.path.exists(classes_dict_path):
                classes_dict_path = None

        alphabet = ALPHABET
        self.alphabet_dict = defaultdict(lambda: 0)
        for i, letter in enumerate(alphabet):
            self.alphabet_dict[letter] = i
        self.alphabet_dict["X"] = 0
        self.files = defaultdict(lambda: defaultdict(list))  # file path by biounit id
        self.loaded = None
        self.dataset_folder = dataset_folder
        self.features_folder = features_folder
        self.cut_edges = cut_edges
        self.mask_residues = mask_residues
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.mask_frac = mask_frac
        self.mask_whole_chains = mask_whole_chains
        self.force_binding_sites_frac = force_binding_sites_frac
        self.mask_all_cdrs = mask_all_cdrs
        self.load_ligands = load_ligands
        self.pyg_graph = pyg_graph
        self.patch_around_mask = patch_around_mask
        self.initial_patch_size = initial_patch_size
        self.antigen_patch_size = antigen_patch_size
        self.mask_sequential = mask_sequential
        self.feature_types = []
        if node_features_type is not None:
            self.feature_types = node_features_type.split("+")
        self.entry_type = entry_type
        self.shuffle_clusters = shuffle_clusters
        self.feature_functions = {
            "sidechain_orientation": self._sidechain,
            "dihedral": self._dihedral,
            "chemical": self._chemical,
            "secondary_structure": self._sse,
            "sidechain_coords": self._sidechain_coords,
        }
        self.feature_functions.update(feature_functions or {})
        if classes_to_exclude is not None and not all(
            [x in ["single_chain", "heteromer", "homomer"] for x in classes_to_exclude]
        ):
            raise ValueError(
                "Invalid class to exclude, choose from 'single_chain', 'heteromer', 'homomer'"
            )

        if debug_file_path is not None:
            self.dataset_folder = os.path.dirname(debug_file_path)
            debug_file_path = os.path.basename(debug_file_path)

        self.main_atom_dict = defaultdict(lambda: None)
        d1to3 = {v: k for k, v in D3TO1.items()}
        for i, letter in enumerate(alphabet):
            if i == 0:
                continue
            self.main_atom_dict[i] = MAIN_ATOMS[d1to3[letter]]

        # create feature folder if it does not exist
        if not os.path.exists(self.features_folder):
            os.makedirs(self.features_folder)

        self.interpolate = interpolate
        # generate the feature files
        print("Processing files...")
        if debug_file_path is None:
            to_process = [
                x for x in os.listdir(dataset_folder) if x.endswith(".pickle")
            ]
        else:
            to_process = [debug_file_path]
        if clustering_dict_path is not None and use_fraction < 1:
            with open(clustering_dict_path, "rb") as f:
                clusters = pickle.load(f)
            keys = sorted(clusters.keys())[: int(len(clusters) * use_fraction)]
            to_process = set()
            for key in keys:
                to_process.update([x[0] for x in clusters[key]])

            file_set = set(os.listdir(dataset_folder))
            to_process = [x for x in to_process if x in file_set]
        if debug:
            to_process = to_process[:1000]
        if self.entry_type == "pair":
            print(
                "Please note that the pair entry type takes longer to process than the other two. The progress bar is not linear because of the varying number of chains per file."
            )
        output_tuples_list = t_map(
            lambda x: self._process(
                x,
                rewrite=rewrite,
                max_length=max_length,
                min_cdr_length=min_cdr_length,
                classes_to_exclude=classes_to_exclude,
            ),
            to_process,
        )
        # save the file names
        for output_tuples in output_tuples_list:
            for id, filename, chain_set in output_tuples:
                for chain in chain_set:
                    self.files[id][chain].append(filename)
        if classes_to_exclude is None:
            classes_to_exclude = []
        classes = None
        if classes_dict_path is not None:
            with open(classes_dict_path, "rb") as f:
                classes = pickle.load(f)
        if clustering_dict_path is not None:
            with open(clustering_dict_path, "rb") as f:
                self.clusters = pickle.load(f)  # list of biounit ids by cluster id
                if classes is None:  # old way of storing class information
                    try:
                        classes = pickle.load(f)
                    except EOFError:
                        pass
        else:
            self.clusters = None
        if classes is None and len(classes_to_exclude) > 0:
            raise ValueError(
                "Classes to exclude are given but no classes dictionary is found, please set classes_dict_path to the path of the classes dictionary"
            )
        to_exclude = set()

        # if classes is not None:
        #     for c in classes_to_exclude:
        #         for key, id_arr in classes.get(c, {}).items():
        #             for id, _ in id_arr:
        #                 to_exclude.add(id)
        if require_antigen or require_light_chain:
            to_exclude.update(
                self._exclude_by_chains(
                    require_antigen,
                    require_light_chain,
                    require_no_light_chain,
                    require_heavy_chain,
                )
            )
        if self.clusters is not None:
            self._exclude_ids_from_clusters(to_exclude)
            self.data = list(self.clusters.keys())
        else:
            self.data = [x for x in self.files.keys() if x not in to_exclude]
        # create a smaller dataset if necessary (if we have clustering it's applied earlier)
        if self.clusters is None and use_fraction < 1:
            self.data = sorted(self.data)[: int(len(self.data) * use_fraction)]
        if load_to_ram:
            print("Loading to RAM...")
            self.loaded = {}
            seen = set()
            for id in self.files:
                for chain, file_list in self.files[id].items():
                    for file in file_list:
                        if file in seen:
                            continue
                        seen.add(file)
                        with open(file, "rb") as f:
                            self.loaded[file] = pickle.load(f)
        sample_file = list(self.files.keys())[0]
        sample_chain = list(self.files[sample_file].keys())[0]
        self.sabdab = "__" in sample_chain
        self.cdr = 0
        self.set_cdr(None)

    def _exclude_ids_from_clusters(self, to_exclude):
        for key in list(self.clusters.keys()):
            cluster_list = []
            for x in self.clusters[key]:
                if x[0] in to_exclude:
                    continue
                id = x[0].split(".")[0]
                chain = x[1]
                if id not in self.files:
                    continue
                if chain not in self.files[id]:
                    continue
                if len(self.files[id][chain]) == 0:
                    continue
                cluster_list.append([id, chain])
            self.clusters[key] = cluster_list
            if len(self.clusters[key]) == 0:
                self.clusters.pop(key)

    def _check_chain_types(self, file):
        chain_types = set()
        with open(file, "rb") as f:
            data = pickle.load(f)
        chains = data["chain_dict"].values()
        for chain in chains:
            chain_mask = data["chain_encoding_all"] == chain
            cdr = data["cdr"][chain_mask]
            cdr_values = cdr.unique()
            if len(cdr_values) == 1:
                chain_types.add("antigen")
            elif CDR_REVERSE["H1"] in cdr_values:
                chain_types.add("heavy")
            elif CDR_REVERSE["L1"] in cdr_values:
                chain_types.add("light")
        return chain_types

    def _exclude_by_class(
        self,
        classes_to_exclude,
    ):
        """Exclude entries that are in the classes to exclude."""
        to_exclude = set()
        for id in self.files:
            for chain in self.files[id]:
                filename = self.files[id][chain][0]
                with open(filename, "rb") as f:
                    data = pickle.load(f)
                if classes_to_exclude in data["classes"]:
                    to_exclude.add(id)
        return to_exclude

    def _exclude_by_chains(
        self,
        require_antigen,
        require_light_chain,
        require_no_light_chain,
        require_heavy_chain,
    ):
        """Exclude entries that do not have an antigen or a light chain."""
        to_exclude = set()
        for id in self.files:
            filename = list(self.files[id].values())[0][
                0
            ]  # assuming entry type is biounit
            chain_types = self._check_chain_types(filename)
            if require_antigen and "antigen" not in chain_types:
                to_exclude.add(id)
            if require_light_chain and "light" not in chain_types:
                to_exclude.add(id)
            if require_no_light_chain and "light" in chain_types:
                to_exclude.add(id)
            if require_heavy_chain and "heavy" not in chain_types:
                to_exclude.add(id)
        return to_exclude

    def _get_masked_sequence(
        self,
        data,
    ):
        """Get the mask for the residues that need to be predicted.

        Depending on the parameters the residues are selected as follows:
        - if `mask_whole_chains` is `True`, the whole chain is masked
        - if `mask_frac` is given, the number of residues to mask is `mask_frac` times the length of the chain,
        - otherwise, the number of residues to mask is sampled uniformly from the range [`lower_limit`, `upper_limit`].

        If `mask_sequential` is `True`, the residues are masked based on the order in the sequence, otherwise a
        spherical mask is applied based on the coordinates.

        If `force_binding_sites_frac` > 0 and `mask_whole_chains` is `False`, in the fraction of cases where a chain
        from a polymer is sampled, the center of the masked region will be forced to be in a binding site.

        Parameters
        ----------
        data : dict
            an entry generated by `ProteinDataset`

        Returns
        -------
        chain_M : torch.Tensor
            a `(B, L)` shaped binary tensor where 1 denotes the part that needs to be predicted and
            0 is everything else

        """
        if "cdr" in data and "cdr_id" in data:
            chain_M = torch.zeros_like(data["cdr"])
            if self.mask_all_cdrs:
                chain_M = data["cdr"] != CDR_REVERSE["-"]
            else:
                chain_M = data["cdr"] == data["cdr_id"]
        else:
            chain_M = torch.zeros_like(data["S"])
            chain_index = data["chain_id"]
            chain_bool = data["chain_encoding_all"] == chain_index

            if self.mask_whole_chains:
                chain_M[chain_bool] = 1
            else:
                chains = torch.unique(data["chain_encoding_all"])
                chain_start = torch.where(chain_bool)[0][0]
                chain = data["X"][chain_bool]
                res_i = None
                interface = []
                non_masked_interface = []
                if len(chains) > 1 and self.force_binding_sites_frac > 0:
                    if random.uniform(0, 1) <= self.force_binding_sites_frac:
                        X_copy = data["X"]

                        i_indices = (chain_bool == 0).nonzero().flatten()  # global
                        j_indices = chain_bool.nonzero().flatten()  # global

                        distances = torch.norm(
                            X_copy[i_indices, 2, :]
                            - X_copy[j_indices, 2, :].unsqueeze(1),
                            dim=-1,
                        ).cpu()
                        close_idx = (
                            np.where(torch.min(distances, dim=1)[0] <= 10)[0]
                            + chain_start.item()
                        )  # global

                        no_mask_idx = (
                            np.where(data["mask"][chain_bool])[0] + chain_start.item()
                        )  # global
                        interface = np.intersect1d(close_idx, j_indices)  # global

                        not_end_mask = np.where(
                            (X_copy[:, 2, :].cpu() == 0).sum(-1) != 3
                        )[0]
                        interface = np.intersect1d(interface, not_end_mask)  # global

                        non_masked_interface = np.intersect1d(interface, no_mask_idx)
                        interpolate = True
                        if len(non_masked_interface) > 0:
                            res_i = non_masked_interface[
                                random.randint(0, len(non_masked_interface) - 1)
                            ]
                        elif len(interface) > 0 and interpolate:
                            res_i = interface[random.randint(0, len(interface) - 1)]
                        else:
                            res_i = no_mask_idx[random.randint(0, len(no_mask_idx) - 1)]
                if res_i is None:
                    non_zero = torch.where(data["mask"] * chain_bool)[0]
                    res_i = non_zero[random.randint(0, len(non_zero) - 1)]
                res_coords = data["X"][res_i, 2, :]
                neighbor_indices = torch.where(data["mask"][chain_bool])[0]
                if self.mask_frac is not None:
                    assert self.mask_frac > 0 and self.mask_frac < 1
                    k = int(len(neighbor_indices) * self.mask_frac)
                else:
                    up = min(
                        self.upper_limit, int(len(neighbor_indices) * 0.5)
                    )  # do not mask more than half of the sequence
                    low = min(up - 1, self.lower_limit)
                    k = random.choice(range(low, up))
                if self.mask_sequential:
                    start = max(1, res_i - chain_start - k // 2)
                    end = min(len(chain) - 1, res_i - chain_start + k // 2)
                    chain_M[chain_start + start : chain_start + end] = 1
                else:
                    dist = torch.norm(
                        chain[neighbor_indices, 2, :] - res_coords.unsqueeze(0), dim=-1
                    )
                    closest_indices = neighbor_indices[
                        torch.topk(dist, k, largest=False)[1]
                    ]
                    chain_M[closest_indices + chain_start] = 1
        return chain_M

    def _dihedral(self, data_entry, chains):
        """Return dihedral angles."""
        return data_entry.dihedral_angles(chains)

    def _sidechain(self, data_entry, chains):
        """Return Sidechain orientation (defined by the 'main atoms' in the `main_atom_dict` dictionary)."""
        return data_entry.sidechain_orientation(chains)

    def _chemical(self, data_entry, chains):
        """Return hemical features (hydropathy, volume, charge, polarity, acceptor/donor)."""
        return data_entry.chemical_features(chains)

    def _sse(self, data_entry, chains):
        """Return secondary structure features."""
        return data_entry.secondary_structure(chains)

    def _sidechain_coords(self, data_entry, chains):
        """Return idechain coordinates."""
        return data_entry.sidechain_coordinates(chains)

    def _process(
        self,
        filename,
        rewrite=False,
        max_length=None,
        min_cdr_length=None,
        classes_to_exclude=None,
    ):
        """Process a proteinflow file and save it as ProteinMPNN features."""
        input_file = os.path.join(self.dataset_folder, filename)
        no_extension_name = filename.split(".")[0]
        data_entry = ProteinEntry.from_pickle(input_file)
        if self.load_ligands:
            ligands = ProteinEntry.retrieve_ligands_from_pickle(input_file)
        if classes_to_exclude is not None:
            if data_entry.get_protein_class() in classes_to_exclude:
                return []
        chains = data_entry.get_chains()
        if self.entry_type == "biounit":
            chain_sets = [chains]
        elif self.entry_type == "chain":
            chain_sets = [[x] for x in chains]
        elif self.entry_type == "pair":
            if len(chains) == 1:
                return []
            chain_sets = list(combinations(chains, 2))
        else:
            raise RuntimeError(
                "Unknown entry type, please choose from ['biounit', 'chain', 'pair']"
            )
        output_names = []
        if self.cut_edges:
            data_entry.cut_missing_edges()
        for chains_i, chain_set in enumerate(chain_sets):
            output_file = os.path.join(
                self.features_folder, no_extension_name + f"_{chains_i}.pickle"
            )
            pass_set = False
            add_name = True
            if os.path.exists(output_file) and not rewrite:
                pass_set = True
                if max_length is not None:
                    if data_entry.get_length(chain_set) > max_length:
                        add_name = False
                if min_cdr_length is not None and data_entry.has_cdr():
                    cdr_length = data_entry.get_cdr_length(chain_set)
                    if not all(
                        [
                            length >= min_cdr_length
                            for length in cdr_length.values()
                            if length > 0
                        ]
                    ):
                        add_name = False
            else:
                if max_length is not None:
                    if data_entry.get_length(chains=chain_set) > max_length:
                        pass_set = True
                        add_name = False
                if min_cdr_length is not None and data_entry.has_cdr():
                    cdr_length = data_entry.get_cdr_length(chain_set)
                    if not all(
                        [
                            length >= min_cdr_length
                            for length in cdr_length.values()
                            if length > 0
                        ]
                    ):
                        pass_set = True
                        add_name = False

                if self.entry_type == "pair":
                    if not data_entry.is_valid_pair(*chain_set):
                        pass_set = True
                        add_name = False
            out = {}
            if add_name:
                cdr_chain_set = set()
                if data_entry.has_cdr():
                    out["cdr"] = torch.tensor(
                        data_entry.get_cdr(chain_set, encode=True)
                    )
                    chain_type_dict = data_entry.get_chain_type_dict(chain_set)
                    out["chain_type_dict"] = chain_type_dict
                    if "heavy" in chain_type_dict:
                        cdr_chain_set.update(
                            [
                                f"{chain_type_dict['heavy']}__{cdr}"
                                for cdr in ["H1", "H2", "H3"]
                            ]
                        )
                    if "light" in chain_type_dict:
                        cdr_chain_set.update(
                            [
                                f"{chain_type_dict['light']}__{cdr}"
                                for cdr in ["L1", "L2", "L3"]
                            ]
                        )
                output_names.append(
                    (
                        os.path.basename(no_extension_name),
                        output_file,
                        chain_set if len(cdr_chain_set) == 0 else cdr_chain_set,
                    )
                )
            if pass_set:
                continue

            if self.interpolate != "none":
                data_entry.interpolate_coords(fill_ends=(self.interpolate == "all"))

            out["pdb_id"] = no_extension_name.split("-")[0]
            out["mask_original"] = torch.tensor(
                data_entry.get_mask(chain_set, original=True)
            )
            out["mask"] = torch.tensor(data_entry.get_mask(chain_set, original=False))
            out["S"] = torch.tensor(data_entry.get_sequence(chain_set, encode=True))
            out["X"] = torch.tensor(data_entry.get_coordinates(chain_set, bb_only=True))
            out["residue_idx"] = torch.tensor(
                data_entry.get_index_array(chain_set, index_bump=100)
            )
            out["chain_encoding_all"] = torch.tensor(
                data_entry.get_chain_id_array(chain_set)
            )
            out["chain_dict"] = data_entry.get_chain_id_dict(chain_set)
            if self.load_ligands and len(ligands) != 0:
                (
                    out["X_ligands"],
                    out["ligand_smiles"],
                    out["ligand_chains"],
                ) = data_entry.get_ligand_features(ligands, chain_set)
            for name in self.feature_types:
                if name not in self.feature_functions:
                    continue
                func = self.feature_functions[name]
                out[name] = torch.tensor(func(data_entry, chain_set))

            with open(output_file, "wb") as f:
                pickle.dump(out, f)

        return output_names

    def set_cdr(self, cdr):
        """Set the CDR to be iterated over (only for SAbDab datasets).

        Parameters
        ----------
        cdr : list | str | None
            The CDR to be iterated over (choose from H1, H2, H3, L1, L2, L3).
            Set to `None` to go back to iterating over all chains.

        """
        if not self.sabdab:
            cdr = None
        if isinstance(cdr, str):
            cdr = [cdr]
        if cdr == self.cdr:
            return
        self.cdr = cdr
        if cdr is None:
            self.indices = list(range(len(self.data)))
        else:
            self.indices = []
            print(f"Setting CDR to {cdr}...")
            for i, data in tqdm(enumerate(self.data)):
                if self.clusters is not None:
                    if data.split("__")[1] in cdr:
                        self.indices.append(i)
                else:
                    add = False
                    for chain in self.files[data]:
                        if chain.split("__")[1] in cdr:
                            add = True
                            break
                    if add:
                        self.indices.append(i)

    def _to_pyg_graph(self, data):
        """Convert a dictionary of data to a PyTorch Geometric graph."""
        from torch_geometric.data import Data

        pyg_data = Data(x=data["X"])
        for key, value in data.items():
            pyg_data[key] = value.unsqueeze(0)
        return pyg_data

    @staticmethod
    def get_anchor_ind(masked_res, mask):
        """Get the indices of the anchor residues.

        Anchor residues are defined as the first and last known residues before and
        after each continuous masked region.

        Parameters
        ----------
        masked_res : torch.Tensor
            A boolean tensor indicating which residues should be predicted
        mask : torch.Tensor
            A boolean tensor indicating which residues are known

        Returns
        -------
        list
            A list of indices of the anchor residues

        """
        anchor_ind = []
        masked_ind = torch.where(masked_res.bool())[0]
        known_ind = torch.where(mask.bool())[0]
        for _, g in groupby(enumerate(masked_ind), lambda x: x[0] - x[1]):
            group = map(itemgetter(1), g)
            group = list(map(int, group))
            start, end = group[0], group[-1]
            start = (
                known_ind[known_ind < start][-1]
                if (known_ind < start).sum() > 0
                else known_ind[0]
            )
            end = (
                known_ind[known_ind > end][0]
                if (known_ind > end).sum() > 0
                else known_ind[-1]
            )
            anchor_ind += [start, end]
        return anchor_ind

    def _get_antibody_mask(self, data):
        """Get a mask for the antibody residues."""
        mask = torch.zeros_like(data["mask"]).bool()
        cdrs = data["cdr"]
        chain_enc = data["chain_encoding_all"]
        for chain_ind in data["chain_dict"].values():
            chain_mask = chain_enc == chain_ind
            chain_cdrs = cdrs[chain_mask]
            if len(torch.unique(chain_cdrs)) > 1:
                mask[chain_mask] = True
        return mask

    def _patch(self, data):
        """Cut the data around the anchor residues."""
        # adapted from diffab
        pos_alpha = data["X"][:, 2]
        if self.mask_whole_chains:
            mask_ = (data["mask"] * data["masked_res"]).bool()
            anchor_points = pos_alpha[mask_].mean(0).unsqueeze(0)
            anchor_ind = []
        else:
            anchor_ind = self.get_anchor_ind(data["masked_res"], data["mask"])
            anchor_points = torch.stack([pos_alpha[ind] for ind in anchor_ind], dim=0)
        dist_anchor = torch.cdist(pos_alpha, anchor_points, p=2).min(dim=1)[0]  # (L, )
        dist_anchor[~data["mask"].bool()] = float("+inf")
        initial_patch_idx = torch.topk(
            dist_anchor,
            k=min(self.initial_patch_size, dist_anchor.size(0)),
            largest=False,
            sorted=True,
        )[
            1
        ]  # (initial_patch_size, )
        patch_mask = data["masked_res"].bool().clone()
        patch_mask[[int(x) for x in anchor_ind]] = True
        patch_mask[initial_patch_idx] = True

        if self.sabdab:
            antibody_mask = self._get_antibody_mask(data)
            antigen_mask = ~antibody_mask
            dist_anchor_antigen = dist_anchor.masked_fill(
                mask=antibody_mask, value=float("+inf")  # Fill antibody with +inf
            )  # (L, )
            antigen_patch_idx = torch.topk(
                dist_anchor_antigen,
                k=min(self.antigen_patch_size, antigen_mask.sum().item()),
                largest=False,
            )[
                1
            ]  # (ag_size, )
            patch_mask[antigen_patch_idx] = True

        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value[patch_mask]
        return data

    def __len__(self):
        """Return the number of clusters or data entries in the dataset."""
        return len(self.indices)

    def __getitem__(self, idx):
        """Return an entry from the dataset.

        If a clusters file is provided, then the idx is the index of the cluster
        and the chain is randomly selected from the cluster. Otherwise, the idx
        is the index of the data entry and the chain is randomly selected from
        the data entry.

        """
        chain_id = None
        cdr = None
        idx = self.indices[idx]
        if self.clusters is None:
            id = self.data[idx]  # data is already filtered by length
            chain_id = random.choice(list(self.files[id].keys()))
            if self.cdr is not None:
                while chain_id.split("__")[1] not in self.cdr:
                    chain_id = random.choice(list(self.files[id].keys()))
        else:
            cluster = self.data[idx]
            id = None
            chain_n = -1
            while (
                id is None or len(self.files[id][chain_id]) == 0
            ):  # some IDs can be filtered out by length
                if self.shuffle_clusters:
                    chain_n = random.randint(0, len(self.clusters[cluster]) - 1)
                else:
                    chain_n += 1
                id, chain_id = self.clusters[cluster][
                    chain_n
                ]  # get id and chain from cluster
        file = random.choice(self.files[id][chain_id])
        if "__" in chain_id:
            chain_id, cdr = chain_id.split("__")
        if self.loaded is None:
            with open(file, "rb") as f:
                try:
                    data = pickle.load(f)
                except EOFError:
                    print("EOFError", file)
                    raise
        else:
            data = deepcopy(self.loaded[file])
        data["chain_id"] = data["chain_dict"][chain_id]
        if cdr is not None:
            data["cdr_id"] = CDR_REVERSE[cdr]
        if self.mask_residues:
            data["masked_res"] = self._get_masked_sequence(data)
        if self.patch_around_mask:
            data = self._patch(data)
        if self.pyg_graph:
            data = self._to_pyg_graph(data)
        return data
