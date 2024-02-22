from typing import Any, Dict

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
from torchmetrics import MeanMetric
from src.utils.rigid_utils import Rigids
from src.utils import losses
from src.models.components.primitives import generate_sinusoidal_encodings


class KestrelLitModule(LightningModule):
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

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
            self,
            pair_feature_net: torch.nn.Module,
            structure_net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            compile: bool,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param pair_feature_net: featurizes the pair representation
        :param structure_net: computes the final structure given pair and single rep.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param compile: whether to compile the models
        """
        super().__init__()

        # this line allows access to init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # The two models that will be tested in Kestrel
        self.pair_feature_net = pair_feature_net
        self.structure_net = structure_net

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(
            self,
            residue_idx: torch.Tensor,
            coordinates: torch.Tensor,
            residue_mask: torch.Tensor
    ) -> Rigids:
        """Perform a forward pass through the model.

        :param residue_idx:
            [*, n_res] a tensor of residue indices
        :param coordinates:
            [*, n_res, 4, 3] a tensor of initial coordinates
        :param residue_mask:
            [*, n_res]

        :return:
            [*, n_res] Rigids object
        """

        batch_size, n_res = residue_mask.shape

        # Pair Features
        pair_repr = self.pair_feature_net(residue_idx=residue_idx,
                                          ca_coordinates=coordinates[:, :, 1, :],
                                          residue_mask=residue_mask)
        # Single rep features as residue index  [*, n_res, c_s]
        single_repr = generate_sinusoidal_encodings(residue_idx, c_s=self.structure_net.c_s)

        # Initialize transforms as identity
        transforms = Rigids.identity((batch_size, n_res))
        transforms = transforms.to(coordinates.float())

        # Apply Structure network
        updated_transforms = self.structure_net(single_repr,
                                                pair_repr,
                                                transforms,
                                                residue_mask)
        return updated_transforms

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def model_step(
            self, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Perform a single model step on a batch of data.

        :param batch:
            a batch of data containing the dictionary returned by ProteinDataModule.

        :return:
            loss tensor
        """
        residue_idx = batch['residue_idx']
        coordinates = batch['X']
        residue_mask = batch['mask']

        pred_frames = self.forward(residue_idx, coordinates, residue_mask)
        pred_positions = pred_frames.get_trans()  # use Ca coordinate values

        # Compute FAPE^2 error
        gt_frames = Rigids.from_3_points(coordinates[:, :, 0, :],  # N
                                         coordinates[:, :, 1, :],  # CA
                                         coordinates[:, :, 2, :])  # C
        gt_positions = gt_frames.get_trans()

        fape = losses.fape_loss(pred_frames=pred_frames,
                                pred_positions=pred_positions,
                                target_frames=gt_frames,
                                target_positions=gt_positions,
                                frames_mask=residue_mask,
                                positions_mask=residue_mask)

        return fape

    def training_step(
            self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
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

    def on_before_zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        """Keeps an eye on weight norms during training."""
        weight_norms = {}
        for name, param in self.named_parameters():
            weight_norms[f"{name}_abs_mean"] = param.abs().mean().item()
        self.log_dict(weight_norms)

    def on_before_optimizer_step(self, optimizer):
        """Keeps an eye on gradient norms during training."""
        norms = grad_norm(self.structure_net, norm_type=2)
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
            self.structure_net = torch.compile(self.structure_net)
            self.pair_feature_net = torch.compile(self.pair_feature_net)

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
    _ = KestrelLitModule(None, None, None, None, False)
