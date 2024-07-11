from typing import Any, Dict

import torch
from torch import nn
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
from torchmetrics import MeanMetric
from src.diffusion.sample import noise_positions, sample_noise_level
from src.diffusion.augmentation import centre_random_augmentation
from src.diffusion.loss import diffusion_loss
from src.utils.geometry.vector import Vec3Array
from src.utils.exponential_moving_average import ExponentialMovingAverage
from src.utils.tensor_utils import tensor_tree_map


class Proteus(nn.Module):
    """Convenience class that consists of a simple feature embedder and diffusion module.
    This is used to make the ProteusLitModule receiving a single nn.Module as x."""

    def __init__(
            self,
            feature_embedder: torch.nn.Module,
            diffusion_module: torch.nn.Module
    ) -> None:
        """
        Args:
            feature_embedder:
                InputFeatureEmbedder to use embed the initial features.
            diffusion_module:
                DiffusionModule to use denoise the noisy atoms."""
        super().__init__()
        self.feature_embedder = torch.compile(feature_embedder)  # TODO: awkward, fix this
        self.diffusion_module = diffusion_module

    def forward(
            self,
            noisy_atoms: Vec3Array,  # (bs, n_atoms)
            timesteps: torch.Tensor,  # (bs, 1)
            features: Dict[str, torch.Tensor],  # x feature dict
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
                x feature dictionary containing the tensors
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
        # This needs to be done manually for DeepSpeed's sake
        dtype = next(self.parameters()).dtype
        for feature in features.keys():
            if features[feature].dtype == torch.float32:
                features[feature] = features[feature].to(dtype=dtype)
        noisy_atoms = noisy_atoms.to_tensor().to(dtype=dtype)
        timesteps = timesteps.to(dtype=dtype)
        token_mask = token_mask.to(dtype=dtype) if token_mask is not None else None
        atom_mask = atom_mask.to(dtype=dtype) if atom_mask is not None else None

        # Initial Features
        init_features = self.feature_embedder(features, atom_mask=atom_mask, token_mask=token_mask)

        # Diffusion module
        denoised_atoms = self.diffusion_module(
            noisy_atoms=noisy_atoms,
            timesteps=timesteps,
            features=features,
            s_inputs=init_features[0],  # TODO: do named accession
            s_trunk=init_features[1],
            z_trunk=init_features[2],
            sd_data=sd_data,
            token_mask=token_mask,
            atom_mask=atom_mask,
        )

        return Vec3Array.from_array(denoised_atoms)


class ProteusLitModule(LightningModule):
    """A small unconditional backbone generation module that is meant as a test fire for AlphaFold3 components."""

    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            matmul_precision: str = "medium",
            compile: bool = False,
            ema_decay: float = 0.999,
    ) -> None:
        """Initialize a ProteusLitModule
        Args:
            optimizer:
                The optimizer to use for training.
            scheduler:
                The learning rate scheduler to use for training.

        TODO: change the initialization to take a config file instead of individual arguments,
         especially for the model since these will have to be initialized separately when loading
         from a checkpoint.
        """
        super().__init__()

        # Simplest diffusion model possible for testing
        self.model = model

        # Maintain an EMA of model parameters
        self.ema = ExponentialMovingAverage(
            model=self.model, decay=ema_decay
        )

        self.cached_weights = None
        self.last_lr_step = -1

        # Save hyperparameters
        self.save_hyperparameters(logger=False, ignore=["model"])

        # for averaging loss across batches  TODO: remove these to reduce clutter
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # Set matmul precision
        torch.set_float32_matmul_precision(matmul_precision)

    def forward(self, *args: Any, **kwargs: Any) -> Vec3Array:
        return self.model(*args, **kwargs)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don'timesteps store results from these checks
        self.val_loss.reset()

    def model_step(
            self, batch: Dict[str, Any]
    ) -> torch.Tensor:
        """Perform a single model step on a batch of data.
        Args:
            batch:
                a batch of data containing the dictionary returned by ProteinDataModule.
        Returns:
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
        timesteps = sample_noise_level((batch_size, 1), device=device, dtype=dtype)  # (bs, 1)
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
        Args:
            batch:
                A batch of data (a tuple) containing the input tensor of images and target labels.
            batch_idx:
                The index of the current batch.
        Returns:
            A tensor of losses between model predictions and targets.
        """
        # Move the EMA to the GPU if not already there
        if self.ema.device != batch["atom_positions"].device:
            self.ema.to(batch["atom_positions"].device)

        loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=False, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

    def on_before_zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        """
        1. Updates the EMA of the model parameters.
        2. Keeps an eye on weight norms during training.
        """
        self.ema.update(self.model)

        # Log weight norms
        weight_norms = {}
        for name, param in self.named_parameters():
            weight_norms[f"{name}_abs_mean"] = param.abs().mean().item()
        self.log_dict(weight_norms)

    def on_before_optimizer_step(self, optimizer):
        """Keeps an eye on gradient norms during training."""
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.
        Args:
            batch: A batch of data (a tuple) containing the x tensor of images and target
                   labels.
            batch_idx:
                The index of the current batch.
        """
        # At the start of validation, load the EMA weights
        if self.cached_weights is None:
            # model.state_dict() contains references to model weights rather
            # than copies. Therefore, we need to clone them before calling
            # load_state_dict().
            clone_param = lambda t: t.detach().clone()
            self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
            self.model.load_state_dict(self.ema.state_dict()["params"])

        loss = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        # Restore the model weights to normal
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        Args:
        batch:
            A batch of data (a tuple) containing the x tensor of images and target
            labels.
        batch_idx:
            The index of the current batch.
        """
        loss = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "name": "AlphaFold3LRScheduler"
                    # "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        """Lightning hook that is called when loading a checkpoint."""
        ema = checkpoint["ema"]
        self.ema.load_state_dict(ema)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        """Lightning hook that is called when saving a checkpoint."""
        checkpoint["ema"] = self.ema.state_dict()


if __name__ == "__main__":
    _ = ProteusLitModule(None, None, None, None)
