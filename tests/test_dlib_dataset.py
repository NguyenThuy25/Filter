import hydra
import omegaconf
import torch
import pyrootutils
# import matplotlib.pyplot as plt
import scipy.io
# import cv2
# import albumentations as A

# from src.data.dlib_datamodule import DlibDataModule

root = pyrootutils.setup_root(__file__, pythonpath=True)
cfg = omegaconf.OmegaConf.load(root / "configs" / "data" / "dlib.yaml")
datamodule = hydra.utils.instantiate(cfg)
datamodule.setup()
# datamodule.draw_batch()

# @pytest.mark.parametrize([32, 128])
# def test_dlib_dataset(cfg: omegaconf.DictConfig):
#     dlib_dataset = hydra.utils.instantiate(cfg)
#     dlib_dataset.setup()
#     DlibDataset = hydra.utils.instantiate()
#     DlibDataset.__len__()
