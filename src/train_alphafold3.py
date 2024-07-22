
# Import libraries
import pytorch_lightning as pl


class AlphaFoldWrapper(pl.LightningModule):
    def __init__(self, config):
        super(AlphaFoldWrapper, self).__init__()
        self.config = config
        self.model = None  # Initialize model here

        self.loss = None  # AlphaFold3 loss

        self.ema = None  # Exponential moving average given decay rate and model

        self.cached_weights = None
        self.last_lr_step = -1
        self.save_hyperparameters()

    def forward(self, batch):
        return self.model(batch)

    def _log(self, loss_breakdown, batch, outputs, train=True):
        # Loop over loss values and log it

        # Compute validation metrics without grad
        pass

    def training_step(self, batch, batch_idx):
        # fetch the ema to the device (wait what, why?)
        # Run the model

        # Multi-chain permutation align the batch

        # Compute loss

        # Log it

        pass

    def on_before_zero_grad(self, *args, **kwargs):
        # Apply EMA to model
        pass

    def validation_step(self, batch, batch_idx):
        # At the start of validation, load the EMA weights

        # Multi-chain permutation align the batch

        # Compute loss

        # Log it

        pass

    def on_validation_epoch_end(self):
        # Restore the model weights to normal
        pass

    def _compute_validation_metrics(self, batch, outputs, superimposition_metrics=False):
        # Metrics
        #
        # Superimposition

        # LDDT Ca score

        # drmsd
        # if computing superimposition metrics:
        # superimpose and compute gdt_ts and gdt_ha
        pass

    def configure_optimizers(self):
        pass

    def on_load_checkpoint(self, checkpoint):
        # Load the EMA model weights
        pass

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()

    def resume_last_lr_step(self, lr_step):
        self.last_lr_step = lr_step


def main():
    # Seed everything

    # Load config (hydra integration)

    # model module

    # Load checkpoint if starting from a checkpoint

    # Compile model if given in args

    # Load the DataModule

    # Prep data and setup

    # Initialize callbacks

    # Initialize logger (WandbLogger)

    # Initialize trainer

    # trainer.fit(model, datamodule, ckpt_path)

    pass


if __name__ == "__main__":
    # Add arguments for training

    # Call main with args
    pass
