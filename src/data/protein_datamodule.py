from typing import Any, Dict, Optional, Tuple, List, Callable

import os
import torch
from torch import nn
from torch.nn import functional as F
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import proteinflow
from torchvision import transforms
from src.data.components.protein_dataset import ProteinDataset


class TransformDataset(torch.utils.data.Dataset):
    """A convenience class that applies arbitrary transforms to torch.utils.data.Dataset objects."""

    def __init__(
            self,
            dataset: Dataset,
            transform: Callable):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return self.transform(sample)

    def __len__(self):
        return self.dataset.__len__()


class Reorder(nn.Module):
    """A transformation that reorders the 3D coordinates of backbone atoms
    from N, C, Ca, O -> N, Ca, C, O."""
    def forward(self, protein_dict):
        # Switch to N, Ca, C, ordering.
        reordered_X = protein_dict['X'].index_select(1, torch.tensor([0, 2, 1, 3]))
        protein_dict['X'] = reordered_X
        return protein_dict


class Cropper(nn.Module):
    """A transformation that crops the protein elements."""

    def __init__(self, crop_size: int = 384):
        super().__init__()
        self.crop_size = crop_size

    def forward(self, protein_dict: dict):
        """Crop the protein
        :param protein_dict: the protein dictionary with the elements
         - 'X': 3D coordinates of N, C, Ca, O, `(total_L, 4, 3)`,
         - 'S': sequence indices (shape `(total_L)`),
         - 'mask': residue mask (0 where coordinates are missing, 1 otherwise; with interpolation 0s are replaced
                   with 1s), (total_L),
         - 'mask_original': residue mask (0 where coordinates are missing, 1 otherwise; not changed with
                              interpolation), `(total_L)`,
         - 'residue_idx': residue indices (from 0 to length of sequence, +100 where chains change),
                            `(total_L)`,
         - 'chain_encoding_all': chain indices, `(total_L)`,
         - 'chain_id': a sampled chain index,
         - 'chain_dict': a dictionary of chain ids (keys are chain ids, e.g. `'A'`, values are the indices
                           used in `'chain_id'` and `'chain_encoding_all'` objects)
        TODO: implement spatial cropping
        """
        n_res = protein_dict['residue_idx'].shape[0]
        n = max(n_res - self.crop_size, 1)
        crop_start = torch.randint(low=0, high=n, size=())
        token_mask = torch.ones_like(protein_dict['mask'], dtype=torch.float32)
        for key, value in protein_dict.items():
            if key == "chain_id" or key == "chain_dict":  # these are not Tensors, so skip
                continue  # omit these from the new dict
            if key == "pdb_id":
                protein_dict[key] = value
                continue  # do not change the pdb_id

            if n_res < self.crop_size:
                padding = torch.zeros((self.crop_size - n_res,) + value.shape[1:], dtype=value.dtype)
                new_value = torch.cat([value, padding], dim=0)
            else:
                new_value = value[crop_start:crop_start + self.crop_size]
            protein_dict[key] = new_value

        # Add token mask
        if n_res < self.crop_size:
            padding = torch.zeros((self.crop_size - n_res,), dtype=torch.float32)
            token_mask = torch.cat([token_mask, padding], dim=0)
        else:
            token_mask = token_mask[crop_start:crop_start + self.crop_size]
        protein_dict['token_mask'] = token_mask
        return protein_dict


class AF3Featurizer(nn.Module):
    """A transformation that featurizes the protein elements to AlphaFold3 features."""
    def __init__(self):
        super().__init__()

    def forward(
            self,
            protein_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Featurize the protein elements to AlphaFold3 features.
        Args:
            protein_dict: the protein dictionary that includes the following elements:
                "X":
                    3D coordinates of N, Ca, C, O, `(total_L, 4, 3)`,
                "S":
                    sequence indices (shape (total_L)),
                "mask":
                    residue mask (0 where coordinates are missing, 1 otherwise; with interpolation 0s are replaced
                    with 1s), (total_L),
                "mask_original":
                    residue mask (0 where coordinates are missing, 1 otherwise; not changed with interpolation),
                    (total_L),
                "residue_idx":
                    residue indices (from 0 to length of sequence, +100 where chains change), (total_L),
                "chain_encoding_all":
                    chain indices, (total_L),
                "chain_id":
                    a sampled chain index,
                "chain_dict":
                    a dictionary of chain ids (keys are chain ids, e.g. 'A', values are the indices
                    used in 'chain_id' and 'chain_encoding_all' objects)
        Returns:
            a dictionary containing the features of AlphaFold3 containing the following elements:
                "residue_index":
                    [n_tokens] Residue number in the token’s original input chain.
                "token_index":
                    [n_tokens] Token number. Increases monotonically; does not restart at 1 for new chains.
                "asym_id":
                    [n_tokens] Unique integer for each distinct chain.
                "entity_id":
                    [n_tokens] Unique integer for each distinct entity.
                "sym_id":
                    [N_tokens] Unique integer within chains of this sequence. E.g. if chains
                    A, B and C share a sequence but D does not, their sym_ids would be [0, 1, 2, 0]
                "ref_pos":
                    [N_atoms, 3] atom positions in the reference conformers, with
                    a random rotation and translation applied. Atom positions in Angstroms.
                "ref_mask":
                    [N_atoms] Mask indicating which atom slots are used in the reference
                    conformer.
                "ref_element":
                    [N_atoms, 128] One-hot encoding of the element atomic number for each atom
                    in the reference conformer, up to atomic number 128.
                "ref_charge":
                    [N_atoms] Charge for each atom in the reference conformer.
                "ref_atom_name_chars":
                    [N_atom, 4, 64] One-hot encoding of the unique atom names in the reference
                    conformer. Each character is encoded as ord(c - 32), and names are padded to
                    length 4.
                "ref_space_uid":
                    [N_atoms] Numerical encoding of the chain id and residue index associated
                    with this reference conformer. Each (chain id, residue index) tuple is assigned
                    an integer on first appearance.
                "atom_to_token":
                    [N_atoms] Token index for each atom in the flat atom representation.
                "atom_exists":
                    [N_atoms] binary mask for atoms, whether atom exists, used for loss masking
                "token_mask":
                    [n_tokens] Mask indicating which tokens are non-padding tokens
                "atom_mask":
                    [N_atoms] Mask indicating which atoms are non-padding atoms
        """
        total_L = protein_dict["residue_idx"].shape[0]  # crop_size
        masks = {
            # Masks
            "token_mask": protein_dict["token_mask"],  # (n_tokens,)
            "atom_mask": protein_dict["token_mask"].unsqueeze(-1).expand(total_L, 4).reshape(total_L * 4)
        }
        af3_features = {
            "residue_index": protein_dict["residue_idx"],
            "token_index": torch.arange(total_L, dtype=torch.float32),
            "asym_id": torch.zeros((total_L,), dtype=torch.float32),
            "entity_id": torch.zeros((total_L,), dtype=torch.float32),
            "sym_id": torch.zeros((total_L,), dtype=torch.float32),
            "ref_pos": protein_dict["X"].reshape(total_L * 4, 3),
            "ref_mask": torch.ones((total_L * 4,), dtype=torch.float32),
            "ref_element": F.one_hot(torch.tensor([7, 6, 6, 8]).unsqueeze(0).expand(total_L, 4).reshape(total_L * 4),
                                     num_classes=128),  # N, C, C, O  atoms repeating in 4s for each residue
            "ref_charge": torch.zeros((total_L * 4,), dtype=torch.float32),
            "ref_atom_name_chars": AF3Featurizer.compute_atom_name_chars(
                ["N", "CA", "C", "O"]).unsqueeze(0).expand(total_L, 4, 4, 64).reshape(total_L * 4, 4, 64),
            "ref_space_uid": protein_dict["residue_idx"].unsqueeze(-1).expand(total_L, 4).reshape(total_L * 4),
            "atom_to_token": torch.arange(total_L).unsqueeze(-1).expand(total_L, 4).reshape(total_L * 4),
            "atom_exists": protein_dict["mask"].unsqueeze(-1).expand(total_L, 4).reshape(total_L * 4) * masks["atom_mask"],
        }
        return af3_features | masks

    @staticmethod
    def compute_atom_name_chars(atom_names: List[str]) -> torch.Tensor:
        """Compute the one-hot encoding of the unique atom names in the reference conformer.
        Each character is encoded as ord(c) - 32 and names are padded to length 4."""
        atom_name_chars = torch.zeros((len(atom_names), 4, 64), dtype=torch.float32)
        for i, atom_name in enumerate(atom_names):
            for j, char in enumerate(atom_name):
                atom_name_chars[i, j, ord(char) - 32] = 1
        return atom_name_chars


class ProteinDataModule(LightningDataModule):
    """`LightningDataModule` for the Protein Data Bank.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
            self,
            data_dir: str = "./data/",
            resolution_thr: float = 3.5,
            min_seq_id: float = 0.3,
            crop_size: int = 384,
            max_length: int = 10_000,
            use_fraction: float = 1.0,
            entry_type: str = "chain",
            classes_to_exclude: Optional[List[str]] = None,
            mask_residues: bool = False,
            lower_limit: int = 15,
            upper_limit: int = 100,
            mask_frac: Optional[float] = None,
            mask_sequential: bool = False,
            mask_whole_chains: bool = False,
            force_binding_sites_frac: float = 0.15,
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            debug: bool = False
    ) -> None:
        """Initialize a `ProteinDataModule`.

        :param resolution_thr: Resolution threshold for PDB structures
        :param min_seq_id: Minimum sequence identity for MMSeq2 clustering
        :param crop_size: The number of residues to crop the proteins to.
        :param max_length: Entries with total length of chains larger than max_length will be disregarded.
        :param use_fraction: the fraction of the clusters to use (first N in alphabetic order)
        :param entry_type: {"biounit", "chain", "pair"} the type of entries to generate ("biounit" for biounit-level
                            complexes, "chain" for chain-level, "pair" for chain-chain pairs (all pairs that are seen
                            in the same biounit and have intersecting coordinate clouds))
        :param classes_to_exclude: a list of classes to exclude from the dataset (select from "single_chains",
                                   "heteromers", "homomers")
        :param mask_residues: if True, the masked residues will be added to the output
        :param lower_limit: the lower limit of the number of residues to mask
        :param upper_limit: the upper limit of the number of residues to mask
        :param mask_frac: if given, the number of residues to mask is mask_frac times the length of the chain
        :param mask_sequential: if True, the masked residues will be neighbors in the sequence; otherwise a geometric
                                mask is applied based on the coordinates.
        :param mask_whole_chains: if True, the whole chain is masked
        :param force_binding_sites_frac: if force_binding_sites_frac > 0 and mask_whole_chains is False, in the
                                         fraction of cases where a chain from a polymer is sampled, the center of
                                         the masked region will be forced to be in a binding site (in PDB datasets).
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        :param debug: In debugging mode or not. Defaults to 'False'
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [Cropper(crop_size=crop_size), Reorder(), AF3Featurizer()]  # crop and reorder
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.other = y).
        """
        data_path = os.path.join(self.hparams.data_dir, "proteinflow_20230102_stable")
        if not os.path.exists(data_path):
            # Download precomputed data, PDB cutoff date 27.02.23
            # This is a dataset with min_res=3.5A, min_len=30, max_len=10_000, min_seq_id=0.3, train/val/test=90/5/5
            os.system("proteinflow download --tag 20230102_stable")

    def setup(self, stage: str = "test") -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size}). "
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        train_folder = os.path.join(self.hparams.data_dir, "proteinflow_20230102_stable/train")
        test_folder = os.path.join(self.hparams.data_dir, "proteinflow_20230102_stable/test")
        val_folder = os.path.join(self.hparams.data_dir, "proteinflow_20230102_stable/valid")

        if stage == "fit" and not self.data_train:
            train_ds = ProteinDataset(train_folder,
                                      max_length=self.hparams.max_length,
                                      use_fraction=self.hparams.use_fraction,
                                      entry_type=self.hparams.entry_type,
                                      classes_to_exclude=self.hparams.classes_to_exclude,
                                      mask_residues=self.hparams.mask_residues,
                                      lower_limit=self.hparams.lower_limit,
                                      upper_limit=self.hparams.upper_limit,
                                      mask_frac=self.hparams.mask_frac,
                                      mask_sequential=self.hparams.mask_sequential,
                                      mask_whole_chains=self.hparams.mask_whole_chains,
                                      force_binding_sites_frac=self.hparams.force_binding_sites_frac,
                                      debug=self.hparams.debug)
            val_ds = ProteinDataset(val_folder,
                                    max_length=self.hparams.max_length,
                                    use_fraction=self.hparams.use_fraction,
                                    entry_type=self.hparams.entry_type,
                                    classes_to_exclude=self.hparams.classes_to_exclude,
                                    mask_residues=self.hparams.mask_residues,
                                    lower_limit=self.hparams.lower_limit,
                                    upper_limit=self.hparams.upper_limit,
                                    mask_frac=self.hparams.mask_frac,
                                    mask_sequential=self.hparams.mask_sequential,
                                    mask_whole_chains=self.hparams.mask_whole_chains,
                                    force_binding_sites_frac=self.hparams.force_binding_sites_frac,
                                    debug=self.hparams.debug)
            self.data_val = TransformDataset(val_ds, transform=self.transforms)
            self.data_train = TransformDataset(train_ds, transform=self.transforms)

        elif stage == "test" and not self.data_test:
            test_ds = ProteinDataset(test_folder,
                                     max_length=self.hparams.max_length,
                                     use_fraction=self.hparams.use_fraction,
                                     entry_type=self.hparams.entry_type,
                                     classes_to_exclude=self.hparams.classes_to_exclude,
                                     mask_residues=self.hparams.mask_residues,
                                     lower_limit=self.hparams.lower_limit,
                                     upper_limit=self.hparams.upper_limit,
                                     mask_frac=self.hparams.mask_frac,
                                     mask_sequential=self.hparams.mask_sequential,
                                     mask_whole_chains=self.hparams.mask_whole_chains,
                                     force_binding_sites_frac=self.hparams.force_binding_sites_frac,
                                     debug=self.hparams.debug)
            self.data_test = TransformDataset(test_ds, transform=self.transforms)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.
        :return: The train dataloader.
        """
        """proteinflow.ProteinLoader(self.data_train,
                                             batch_size=self.batch_size_per_device,
                                             num_workers=self.hparams.num_workers,
                                             pin_memory=self.hparams.pin_memory)"""
        return DataLoader(self.data_train,
                          batch_size=self.batch_size_per_device,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        """
        proteinflow.ProteinLoader(self.data_val,
                                         shuffle_batches=False,
                                         batch_size=self.batch_size_per_device,
                                         num_workers=self.hparams.num_workers,
                                         pin_memory=self.hparams.pin_memory)
        """
        return DataLoader(self.data_val,
                          shuffle=False,
                          batch_size=self.batch_size_per_device,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        """proteinflow.ProteinLoader(self.data_test,
                                         shuffle_batches=False,
                                         batch_size=self.batch_size_per_device,
                                         num_workers=self.hparams.num_workers,
                                         pin_memory=self.hparams.pin_memory)"""
        return DataLoader(self.data_test,
                          shuffle=False,
                          batch_size=self.batch_size_per_device,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = ProteinDataModule()
