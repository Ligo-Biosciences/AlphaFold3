# Copyright 2024 Ligo Biosciences Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Optional
import torch
from torch import nn
import hydra
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
from torchmetrics import MeanMetric
from src.utils.exponential_moving_average import ExponentialMovingAverage
from src.utils.tensor_utils import tensor_tree_map
from src.utils.loss import ProteusLoss
from einops import rearrange


class Proteus(nn.Module):
    """Convenience class that consists of a simple feature embedder and diffusion module.
    This is used to make the ProteusLitModule receiving a single nn.Module as input."""

    def __init__(
            self,
            feature_embedder: torch.nn.Module,
            diffusion_module: torch.nn.Module
    ):
        """
        Args:
            feature_embedder:
                InputFeatureEmbedder to use embed the initial features.
            diffusion_module:
                DiffusionModule to use denoise the noisy atoms."""
        super().__init__()
        self.feature_embedder = feature_embedder
        self.diffusion_module = diffusion_module

    def forward(
            self,
            gt_atoms: torch.Tensor,  # (bs, n_atoms)
            features: Dict[str, torch.Tensor],  # input feature dict
            samples_per_trunk: int,
            use_deepspeed_evo_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Perform a forward pass through the model.
        Args:
            gt_atoms:
                vector of ground truth atom positions (bs, n_atoms)
            features:
                input feature dictionary containing the tensors
            samples_per_trunk:
                Number of samples to per trunk conditioning.
            use_deepspeed_evo_attention:
                Whether to use Deepspeed's optimized kernels.

        Returns:
            [*, n_atoms] The denoised positions of the atoms
        """
        atom_mask = features["atom_mask"]
        token_mask = features["token_mask"]

        # Initial Features
        s_inputs, s_trunk, z_trunk = self.feature_embedder(features, atom_mask, token_mask)

        # Diffusion module
        outputs = self.diffusion_module.train_step(
            ground_truth_atoms=gt_atoms,
            features=features,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            samples_per_trunk=samples_per_trunk,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention
        )
        return outputs


class ProteusLitModule(LightningModule):
    """A small unconditional backbone generation module that is meant as a test fire for AlphaFold3 components."""

    def __init__(self, config):
        """Initialize a ProteusLitModule"""
        super().__init__()
        self.config = config

        # Simplest diffusion model possible for testing
        self.model = hydra.utils.instantiate(config.model)  # model

        # Maintain an EMA of model parameters
        self.ema = ExponentialMovingAverage(
            model=self.model, decay=config.ema_decay
        )
        self.cached_weights = None


        self.last_lr_step = -1
        # Save hyperparameters
        self.save_hyperparameters(logger=False)

        # for averaging loss across batches  TODO: remove these to reduce clutter
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.loss_fn = ProteusLoss(config.loss)

        # Set matmul precision
        torch.set_float32_matmul_precision(config.matmul_precision)

    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        return self.model(*args, **kwargs)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

        # Log gradients, parameter histograms, and model topology
        self.logger.watch(self.model, log="all")

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
        batch = reshape_features(batch)  # temporary
        # Remove the recycling dimension
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        # Run the model
        outputs = self.forward(
            gt_atoms=batch["all_atom_positions"],
            features=batch,
            samples_per_trunk=self.config.globals.samples_per_trunk,
            use_deepspeed_evo_attention=self.config.globals.use_deepspeed_evo_attention,
        )
        # Flatten the S to be incorporated into the batch dimension
        # TODO: this is temporary, delete and replace with arbitrary batch dims handling
        outputs["augmented_gt_atoms"] = rearrange(
            outputs["augmented_gt_atoms"], 'b s n c -> (b s) n c'
        )
        outputs["denoised_atoms"] = rearrange(
            outputs["denoised_atoms"], 'b s n c -> (b s) n c'
        )
        outputs["timesteps"] = rearrange(
            outputs["timesteps"], 'b s o -> (b s) o'
        )
        # Compute loss
        loss = self.loss_fn(outputs, batch, _return_breakdown=False)
        return loss

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

        loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=False, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set."""

        loss = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set."""
        loss = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict."""
        if self.config.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self):
        partial_optimizer = hydra.utils.instantiate(self.config.optimizer)
        partial_scheduler = hydra.utils.instantiate(self.config.scheduler)
        optimizer = partial_optimizer(self.trainer.model.parameters())
        scheduler = partial_scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": "AlphaFold3LRScheduler"
                # "frequency": 1,
            },
        }
    
    def on_before_zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Keeps an eye on weight norms during training.
        """
        # Log weight norms
        weight_norms = {}
        for name, param in self.named_parameters():
            weight_norms[f"{name}_abs_mean"] = param.abs().mean().item()
        self.log_dict(weight_norms)

    def on_before_optimizer_step(self, optimizer):
        """Keeps an eye on gradient norms during training."""
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Update EMA after each training batch
        self.ema.update(self.model)
    
    def on_train_batch_start(self, batch: Any, batch_idx: int):
        # Fetch the EMA weights to the device
        if self.ema.device != batch["residue_index"].device:
            self.ema.to(batch["residue_index"].device)

    def on_validation_epoch_start(self):
        # At the start of validation, load the EMA weights
        if self.cached_weights is None:
            # model.state_dict() contains references to model weights rather
            # than copies. Therefore, we need to clone them before calling 
            # load_state_dict().
            clone_param = lambda t: t.detach().clone()
            self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
            self.model.load_state_dict(self.ema.params)

    def on_validation_epoch_end(self):
        # Restore original model weights
        if self.cached_weights is not None:
            self.model.load_state_dict(self.cached_weights)
            self.cached_weights = None

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        """Lightning hook that is called when loading a checkpoint."""
        ema = checkpoint["ema"]
        self.ema.load_state_dict(ema)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        """Lightning hook that is called when saving a checkpoint."""
        checkpoint["ema"] = self.ema.state_dict()


def reshape_features(batch):
    """Temporary function that converts the features in the
    batch to the correct shapes for the model. Assumes only 4 backbone atoms per residue."""
    bs, n_res, _, n_cycle = batch["ref_mask"].shape
    batch["all_atom_positions"] = batch["all_atom_positions"].reshape(-1, n_res * 4, 3, n_cycle)
    batch["ref_pos"] = batch["ref_pos"].reshape(-1, n_res * 4, 3, n_cycle)
    batch["ref_mask"] = batch["ref_mask"].reshape(-1, n_res * 4, n_cycle)
    batch["ref_element"] = batch["ref_element"].reshape(-1, n_res * 4, 4, n_cycle)
    batch["ref_charge"] = batch["ref_charge"].reshape(-1, n_res * 4, n_cycle)
    batch["ref_atom_name_chars"] = batch["ref_atom_name_chars"].reshape(-1, n_res * 4, 4, n_cycle)
    batch["ref_space_uid"] = batch["ref_space_uid"].reshape(-1, n_res * 4, n_cycle)
    batch["atom_to_token"] = batch["atom_to_token"].reshape(-1, n_res * 4, n_cycle)
    batch["atom_exists"] = batch["atom_exists"].reshape(-1, n_res * 4, n_cycle)
    batch["atom_mask"] = batch["atom_mask"].reshape(-1, n_res * 4, n_cycle)
    return batch
