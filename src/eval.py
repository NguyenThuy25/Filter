from typing import List, Tuple

import hydra
import numpy as np
import pyrootutils
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
import torch
from PIL import Image


import torchvision
import cv2

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.dlib_datamodule import TransformDataset  # noqa: E402
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    print("test1")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    print("test2")
    metric_dict = trainer.callback_metrics

    # for predictions use trainer.predict(...)
    log.info("Starting predictions!")
    # predictions = trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    # print("predictions:", len(predictions), type(predictions[0]), predictions[0].shape)
    # for pred, (bx, by) in zip(predictions, datamodule.predict_dataloader()):
    #     print("pred:", pred.shape, "batch:", bx.shape, by.shape)
    #     annotated_image = TransformDataset.annotate_tensor(bx, pred)
    #     print("output_path:", cfg.paths.output_dir + "/eval_result.png")
    #     torchvision.utils.save_image(annotated_image, cfg.paths.output_dir + "/eval_result.png")
    #     break
    img = Image.open('/Users/admin/Downloads/Work/filter_project-main/data/ibug_300W_large_face_landmark_dataset/lfpw/trainset/image_0001.png').convert('RGB')
    img_with_kps = eval_img(img, model)
    print(type(img_with_kps))

    # _ = cv2.imwrite('eval_result.png', img_with_kps)
    torchvision.utils.save_image(img_with_kps, "eval_result.png")
    return metric_dict, object_dict

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)

def eval_img(img, model):
    transform = Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    

    input_image = np.array(img)
    print(input_image.shape)
    input_tensor = transform(image=input_image)
    input_tensor = input_tensor['image'].unsqueeze(0)
    # print(input_tensor.size())
    # input_tensor = cv2.cvtColor(input_tensor.numpy(), cv2.COLOR_RGB2BGR)
    # _ = cv2.imwrite('eval_result3.png', input_tensor)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    # # prediction = torch.argmax(output_tensor, dim=1)
    print(input_tensor.shape, output_tensor.shape) #1 3 224 224, 1 68 2
    img_with_kps = TransformDataset.annotate_batch(input_tensor, output_tensor)
    # cv2.imshow('image', cv2.imread('data/ibug_tiny/helen/trainset/146827737_1.jpg'))
    
    return img_with_kps
if __name__ == "__main__":
    main()
