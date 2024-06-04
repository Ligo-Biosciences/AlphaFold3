from typing import Any, Dict

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
from torchmetrics import MeanMetric
from src.utils.geometry.vector import Vec3Array
from src.diffusion.sample import noise_positions, sample_noise_level
from src.diffusion.augmentation import centre_random_augmentation
from src.diffusion.loss import diffusion_loss
from torch.utils.checkpoint import checkpoint


class ProteusLitModule(LightningModule):
    """A regression-based module that is meant for smaller subtasks.
    It will be used to test the training of model components.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```
    """

    def __init__(
            self,
            feature_embedder: torch.nn.Module,
            diffusion_module: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            compile: bool,
    ) -> None:
        """Initialize a KestrelLitModule
        Args:
            feature_embedder:
                InputFeatureEmbedder to use embed the initial features.
            diffusion_module:
                DiffusionModule to use denoise the noisy atoms.
            optimizer:
                The optimizer to use for training.
            scheduler:
                The learning rate scheduler to use for training.
            compile:
                whether to compile the models
        """
        super().__init__()

        # this line allows access to init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["feature_embedder", "diffusion_module"])

        # Simplest diffusion model possible for testing
        self.feature_embedder = feature_embedder
        self.diffusion_module = diffusion_module

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(
            self,
            noisy_atoms: Vec3Array,  # (bs, n_atoms)
            timesteps: torch.Tensor,  # (bs, 1)
            features: Dict[str, torch.Tensor],  # input feature dict
            sd_data: float = 16.0,
            token_mask: torch.Tensor = None,  # (bs, n_tokens)
            atom_mask: torch.Tensor = None  # (bs, n_atoms)
    ) -> Vec3Array:
        """Perform a forward pass through the model.
        Args:
            noisy_atoms:
                vector of noisy atom positions (bs, n_atoms)
            timesteps:
                tensor of timesteps (bs, 1)
            features:
                input feature dictionary containing the tensors
            sd_data:
                Scaling factor for the timesteps before fourier embedding
            token_mask:
                [*, N_tokens] binary mask for tokens, whether token is present (not padding)
            atom_mask:
                [*, N_atoms] binary mask for atoms, whether atom is present (will still be 1.0 if
                atom is missing from the crystal structure, only 0.0 for padding)

        Returns:
            [*, n_atoms] The denoised positions of the atoms
        """

        # Initial Features
        init_features = self.feature_embedder(features, atom_mask=atom_mask, token_mask=token_mask)

        # Evoformer Block
        denoised_atoms = self.diffusion_module(
            noisy_atoms=noisy_atoms,
            timesteps=timesteps,
            features=features,
            s_inputs=init_features.s_inputs,
            s_trunk=init_features.s_trunk,
            z_trunk=init_features.z_trunk,
            sd_data=sd_data,
            token_mask=token_mask,
            atom_mask=atom_mask
        )

        return denoised_atoms

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don'timesteps store results from these checks
        self.val_loss.reset()

    def model_step(
            self, batch: Dict[str, Any]
    ) -> torch.Tensor:
        """Perform a single model step on a batch of data.

        :param batch:
            a batch of data containing the dictionary returned by ProteinDataModule.

        :return:
            loss tensor
        """

        # Record shape and device
        coordinates = batch["atom_positions"]  # (bs, n_atoms, 3)
        batch_size, n_atoms, _ = coordinates.shape
        device, dtype = coordinates.device, coordinates.dtype

        # Convert to Vec3Array
        atom_positions = Vec3Array.from_array(coordinates)  # (bs, n_atoms)

        # Centre random augmentation
        atom_positions = centre_random_augmentation(atom_positions)

        # Sample timesteps and noise atoms
        timesteps = sample_noise_level(torch.randn(batch_size, 1, device=device, dtype=dtype))  # (bs, 1)
        noisy_positions = noise_positions(atom_positions, timesteps)

        # Run the model
        denoised_positions = self.forward(noisy_positions,
                                          timesteps,
                                          features=batch["features"],
                                          token_mask=batch["token_mask"],
                                          atom_mask=batch["atom_mask"])
        atom_nucleic_acid = batch["features"]["ref_charge"]  # zeros of shape (bs, n_atoms)
        per_example_losses = diffusion_loss(denoised_positions,
                                            atom_positions,
                                            timesteps,
                                            atom_is_rna=atom_nucleic_acid,
                                            atom_is_dna=atom_nucleic_acid,
                                            weights=torch.ones_like(atom_nucleic_acid, device=device, dtype=dtype),
                                            mask=batch["atom_mask"])
        return torch.mean(per_example_losses)

    def training_step(
            self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss = self.model_step(batch)  # model step with grad checkpointing

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=False, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

    def on_before_zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        """Keeps an eye on weight norms during training."""
        weight_norms = {}
        for name, param in self.named_parameters():
            weight_norms[f"{name}_abs_mean"] = param.abs().mean().item()
        self.log_dict(weight_norms)

    def on_before_optimizer_step(self, optimizer):
        """Keeps an eye on gradient norms during training."""
        norms = grad_norm(self.diffusion_module, norm_type=2)
        self.log_dict(norms)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # TODO: add RMSD metric for validation

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        # acc = self.val_acc.compute()  # get current val acc
        # self.val_acc_best(acc)  # update best so far val loss
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        # self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        # TODO: log best validation loss and RMSD metrics
        pass

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        # TODO: add RMSD metric for test

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # TODO: log best test loss and RMSD metrics
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.diffusion_module = torch.compile(self.diffusion_module)
            self.feature_embedder = torch.compile(self.feature_embedder)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = ProteusLitModule(None, None, None, None, compile=False)
