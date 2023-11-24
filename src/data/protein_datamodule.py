from typing import Any, Dict, Optional, Tuple, List, Callable

import os
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import proteinflow
from torchvision import transforms


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


class Reorder(torch.nn.Module):
    """A transformation that reorders the 3D coordinates of backbone atoms
    from N, C, Ca, O -> N, Ca, C, O."""
    def forward(self, protein_dict):
        # Switch to N, Ca, C, ordering.
        reordered_X = protein_dict['X'].index_select(1, torch.tensor([0, 2, 1, 3]))
        protein_dict['X'] = reordered_X
        return protein_dict


class Cropper(torch.nn.Module):
    """A transformation that crops the protein elements."""

    def __init__(self, crop_size: int = 384):
        super().__init__()
        self.crop_size = crop_size

    def forward(self, protein_dict: dict):
        """Crop the protein
        :param protein_dict: the protein dictionary with the elements
         - `'X'`: 3D coordinates of N, C, Ca, O, `(total_L, 4, 3)`,
         - `'S'`: sequence indices (shape `(total_L)`),
         - `'mask'`: residue mask (0 where coordinates are missing, 1 otherwise; with interpolation 0s are
                     replaced with 1s), `(total_L)`,
         - `'mask_original'`: residue mask (0 where coordinates are missing, 1 otherwise; not changed with
                              interpolation), `(total_L)`,
         - `'residue_idx'`: residue indices (from 0 to length of sequence, +100 where chains change),
                            `(total_L)`,
         - `'chain_encoding_all'`: chain indices, `(total_L)`,
         - `'chain_id`': a sampled chain index,
         - `'chain_dict'`: a dictionary of chain ids (keys are chain ids, e.g. `'A'`, values are the indices
                           used in `'chain_id'` and `'chain_encoding_all'` objects)
        """
        n_res = protein_dict['residue_idx'].shape[0]
        n = max(n_res - self.crop_size, 1)
        crop_start = torch.randint(low=0, high=n, size=())
        new_protein_dict = {}
        for key, value in protein_dict.items():
            if key == "chain_id" or key == "chain_dict":  # these are not Tensors, so skip
                continue  # omit these from the new dict
            new_protein_dict[key] = value[crop_start:crop_start + self.crop_size]
        return new_protein_dict


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
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [Cropper(crop_size=crop_size), Reorder()]  # crop and reorder
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

        Do not use it to assign state (self.x = y).
        """
        # Download precomputed data, PDB cutoff date 27.02.23
        # This is a dataset with min_res=3.5A, min_len=30, max_len=10_000, min_seq_id=0.3, train/val/test=90/5/5
        os.system("proteinflow download --tag 20230102_stable")

    def setup(self, stage: Optional[str] = None, debug: bool = False) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        :param debug: debugging mode
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
        if debug:
            # only load the test dataset if in debug mode
            test_dataset = proteinflow.ProteinDataset(test_folder,
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
                                                      force_binding_sites_frac=self.hparams.force_binding_sites_frac)
            self.data_test = TransformDataset(test_dataset, transform=self.transforms)
            self.data_test = test_dataset
        else:
            if not self.data_train and not self.data_val and not self.data_test:
                train_dataset = proteinflow.ProteinDataset(train_folder,
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
                                                           force_binding_sites_frac=self.hparams.force_binding_sites_frac)
                test_dataset = proteinflow.ProteinDataset(test_folder,
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
                                                          force_binding_sites_frac=self.hparams.force_binding_sites_frac)
                val_dataset = proteinflow.ProteinDataset(val_folder,
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
                                                         force_binding_sites_frac=self.hparams.force_binding_sites_frac)

                # Apply transforms
                self.data_train = TransformDataset(train_dataset, transform=self.transforms)
                self.data_val = TransformDataset(val_dataset, transform=self.transforms)
                self.data_test = TransformDataset(test_dataset, transform=self.transforms)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.
        :return: The train dataloader.
        """
        return proteinflow.ProteinLoader(self.data_train,
                                         batch_size=self.batch_size_per_device,
                                         num_workers=self.hparams.num_workers,
                                         pin_memory=self.hparams.pin_memory)

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return proteinflow.ProteinLoader(self.data_val,
                                         batch_size=self.batch_size_per_device,
                                         num_workers=self.hparams.num_workers,
                                         pin_memory=self.hparams.pin_memory)

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return proteinflow.ProteinLoader(self.data_test,
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
