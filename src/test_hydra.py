import hydra
from omegaconf import OmegaConf
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# from src.models.model import AlphaFold3  # from src.models.model import AlphaFold3
from src.data.data_modules import OpenFoldDataModule


@hydra.main(version_base=None, config_path="../configs/data", config_name="erebor.yaml")
def test_hydra(cfg):
    print(OmegaConf.to_yaml(cfg))

    # Initialize the model
    # model = AlphaFold3(cfg)
    # print(model)
    # Initialize the data module
    datamodule = OpenFoldDataModule(cfg)  # hydra.utils.instantiate(cfg)
    print(datamodule)


if __name__ == "__main__":
    test_hydra()
