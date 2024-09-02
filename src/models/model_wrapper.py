import hydra
import lightning as L
import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
from src.utils.tensor_utils import tensor_tree_map
from typing import Any, Dict
from src.models.model import AlphaFold3
from src.utils.loss import AlphaFold3Loss
from src.utils.exponential_moving_average import ExponentialMovingAverage
from einops import rearrange
from src.common import residue_constants
from src.utils.superimposition import superimpose
from src.utils.validation_metrics import (
    drmsd, gdt_ts, gdt_ha, lddt
)


class AlphaFoldWrapper(LightningModule):
    def __init__(self, config):
        super(AlphaFoldWrapper, self).__init__()
        self.config = config
        self.globals = self.config.globals
        self.model = AlphaFold3(config)

        self.loss = AlphaFold3Loss(config.loss)

        self.ema = ExponentialMovingAverage(
            model=self.model, decay=config.ema_decay
        )
        self.cached_weights = None

        self.cached_weights = None
        self.last_lr_step = -1
        self.save_hyperparameters()

        # Set matmul precision
        torch.set_float32_matmul_precision(self.globals.matmul_precision)

    def forward(self, batch, training=True):
        return self.model(batch, train=training)

    def _log(self, batch, outputs, loss_breakdown=None, train=True):
        # Loop over loss values and log it
        phase = "train" if train else "val"
        if loss_breakdown is not None:
            for loss_name, indiv_loss in loss_breakdown.items():
                self.log(
                    f"{phase}/{loss_name}",
                    indiv_loss,
                    prog_bar=(loss_name == 'loss'),
                    on_step=train, on_epoch=(not train), logger=True, sync_dist=False,
                )

        # Compute validation metrics
        other_metrics = self._compute_validation_metrics(
            batch,
            outputs,
            superimposition_metrics=True  # (not train)
        )

        for k, v in other_metrics.items():
            self.log(
                f"{phase}/{k}",
                torch.mean(v),
                prog_bar=(k == 'loss'),
                on_step=train, on_epoch=True, logger=True, sync_dist=True,
            )

    def training_step(self, batch, batch_idx):
        batch = reshape_features(batch)  # temporary

        # Run the model
        outputs = self.forward(batch, training=True)

        # Remove the recycling dimension
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        # For multimer, multichain permutation align the batch

        # Flatten the S to be incorporated into the batch dimension
        # TODO: this is temporary, will be removed once the data pipeline is better written
        outputs["augmented_gt_atoms"] = rearrange(
            outputs["augmented_gt_atoms"], 'b s n c -> (b s) n c'
        )
        outputs["denoised_atoms"] = rearrange(
            outputs["denoised_atoms"], 'b s n c -> (b s) n c'
        )
        outputs["timesteps"] = rearrange(
            outputs["timesteps"], 'b s o -> (b s) o'
        )
        # Expand atom_exists to be of shape (bs * samples_per_trunk, n_atoms)
        samples_per_trunk = outputs["timesteps"].shape[0] // batch["atom_exists"].shape[0]
        expand_batch = lambda tensor: tensor.repeat_interleave(samples_per_trunk, dim=0)
        batch["atom_exists"] = expand_batch(batch["atom_exists"])

        # Compute loss
        loss, loss_breakdown = self.loss(
            outputs, batch, _return_breakdown=True
        )

        # Log loss and validation metrics
        self._log(
            loss_breakdown=loss_breakdown,
            batch=batch,
            outputs=outputs,
            train=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        batch = reshape_features(batch)  # temporary

        # Run the model
        outputs = self.forward(batch, training=False)
        batch = tensor_tree_map(lambda t: t[..., -1], batch)  # Remove recycling dimension

        # For multimer, multichain permutation align the batch

        # Compute and log validation metrics
        self._log(loss_breakdown=None, batch=batch, outputs=outputs, train=False)

    def _compute_validation_metrics(
            self,
            batch,
            outputs,
            superimposition_metrics=False
    ):
        """Compute validation metrics for the model."""
        with torch.no_grad():
            batch_size, n_tokens = batch["token_index"].shape
            metrics = {}

            gt_coords = batch["all_atom_positions"]  # (bs, n_atoms, 3) gt_atoms after augmentation
            pred_coords = outputs["sampled_positions"].squeeze(-3)  # remove S dimension (bs, 1, n_atoms, 3)
            all_atom_mask = batch["atom_mask"]  # (bs, n_atoms)

            # Center the gt_coords
            gt_coords = gt_coords - torch.mean(gt_coords, dim=-2, keepdim=True)

            gt_coords_masked = gt_coords * all_atom_mask[..., None]
            pred_coords_masked = pred_coords * all_atom_mask[..., None]

            # Reshape to backbone atom format (temporary, will switch to more general representation)
            gt_coords_masked = gt_coords_masked.reshape(batch_size, n_tokens, 4, 3)
            pred_coords_masked = pred_coords_masked.reshape(batch_size, n_tokens, 4, 3)
            all_atom_mask = all_atom_mask.reshape(batch_size, n_tokens, 4)

            ca_pos = residue_constants.atom_order["CA"]
            gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :]
            pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
            all_atom_mask_ca = all_atom_mask[..., ca_pos]

            # lddt_ca_score = lddt(
            #    all_atom_pred_pos=pred_coords_masked_ca,
            #    all_atom_positions=gt_coords_masked_ca,
            #    all_atom_mask=all_atom_mask_ca,
            #    eps=self.config.globals.eps,
            #    per_residue=False
            # )
            # metrics["lddt_ca"] = lddt_ca_score

            # drmsd
            drmsd_ca_score = drmsd(
                pred_coords_masked_ca,
                gt_coords_masked_ca,
                mask=all_atom_mask_ca,  # still required here to compute n
            )
            metrics["drmsd_ca"] = drmsd_ca_score

            if superimposition_metrics:
                superimposed_pred, alignment_rmsd = superimpose(
                    gt_coords_masked_ca, pred_coords_masked_ca, all_atom_mask_ca,
                )
                gdt_ts_score = gdt_ts(
                    superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
                )
                gdt_ha_score = gdt_ha(
                    superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
                )

                metrics["alignment_rmsd"] = alignment_rmsd
                metrics["gdt_ts"] = gdt_ts_score
                metrics["gdt_ha"] = gdt_ha_score

            return metrics

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

    # def on_before_optimizer_step(self, optimizer):
    #    """Keeps an eye on gradient norms during training."""
    #    norms = grad_norm(self.model, norm_type=2)
    #    self.log_dict(norms)

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

    def resume_last_lr_step(self, lr_step):
        self.last_lr_step = lr_step


def reshape_features(batch):
    """Temporary function that converts the features in the
    batch to the correct shapes for the model. Assumes only 4 backbone atoms per residue.
    This will be deleted once the data pipeline is more mature.
    """
    bs, n_res, _, n_cycle = batch["ref_mask"].shape
    n_atoms = n_res * 4

    def reshape_feature(feature, *dims):
        return batch[feature].reshape(bs, *dims, n_cycle)

    # Reshape atom-wise features
    batch.update({
        "all_atom_positions": reshape_feature("all_atom_positions", n_atoms, 3),
        "ref_pos": reshape_feature("ref_pos", n_atoms, 3),
        "ref_mask": reshape_feature("ref_mask", n_atoms),
        "ref_element": reshape_feature("ref_element", n_atoms, 4),
        "ref_charge": reshape_feature("ref_charge", n_atoms),
        "ref_atom_name_chars": reshape_feature("ref_atom_name_chars", n_atoms, 4),
        "ref_space_uid": reshape_feature("ref_space_uid", n_atoms),
        "atom_exists": reshape_feature("all_atom_mask", n_atoms),
        "atom_mask": reshape_feature("all_atom_mask", n_atoms),
    })

    # Rename some features
    batch["token_mask"] = batch["seq_mask"]
    batch["token_index"] = batch["residue_index"]

    # Add assembly features
    for feature in ["asym_id", "entity_id", "sym_id"]:
        batch[feature] = torch.zeros_like(batch["seq_mask"])

    # Remove gt_features key
    batch.pop("gt_features", None)

    # Compute and add atom_to_token
    atom_to_token = torch.arange(n_res).unsqueeze(-1).expand(n_res, 4)
    batch["atom_to_token"] = atom_to_token[None, ..., None].expand(bs, n_res, 4, n_cycle).reshape(bs, n_atoms, n_cycle).to(batch["ref_mask"].device)

    return batch
