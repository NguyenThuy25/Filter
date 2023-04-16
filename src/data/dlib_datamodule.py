import os
from PIL import Image
from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from src.data.dlib_dataset import DlibDataset, TransformDataset
import matplotlib.pyplot as plt
from albumentations import Compose


class DlibDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_train: DlibDataset,
        data_test: DlibDataset,
        data_dir: str = "data/ibug_300W_large_face_landmark_dataset",
        train_val_test_split = [5_666, 1_0000],
        transform_train: Optional[Compose] = None,
        transform_val: Optional[Compose] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        # self.transforms = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        # )
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass
        # MNIST(self.hparams.data_dir, train=True, download=True)
        # MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            data_train = self.hparams.data_train(data_dir=self.hparams.data_dir)
            data_test = self.hparams.data_test(data_dir=self.hparams.data_dir)
            # print(data_train.__len__())
            data_train, data_val = random_split(
                dataset = data_train,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )
            self.data_train = TransformDataset(data_train, self.hparams.transform_train)
            self.data_test = TransformDataset(data_test, self.hparams.transform_val)
            self.data_val = TransformDataset(data_test, self.hparams.transform_val)
            # print(self.data_train)
            # print(self.data_val)
            # print(self.data_test)



    def train_dataloader(self):
        # print (len(self.data_train))
        # import IPython ; IPython.embed()
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
    
    def draw_batch(self):
        batch = next(iter(self.train_dataloader()))
        fig = plt.figure(figsize=(8,8))
        key_points = batch['keypoints']

        for i in range(len(batch['image'])):
            ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
            image = batch['image'][i]
            for j in range(68):
                plt.scatter((key_points[i][j][0] + 0.5) *224, (key_points[i][j][1]+0.5)*224, s=10, marker='.', c='r')

            plt.imshow(image)
        plt.show()
    

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils
    from omegaconf import DictConfig
    from tqdm import tqdm

    def test_dataset(cfg: DictConfig):
        dataset: DlibDataset = hydra.utils.instantiate(cfg.data_train)
        dataset = dataset(data_dir=cfg.data_dir)
        print("dataset", len(dataset))
        img, kp = dataset.__getitem__(0)
        
        print("img", img.size, "kp", kp.shape)
        dataset.annotate_image(img, kp)
    
    def test_datamodule(cfg: Dict):
        datamodule: DlibDataModule = hydra.utils.instantiate(cfg)
        # datamodule.prepare_data()
        datamodule.setup()
        loader = datamodule.train_dataloader()
        x, y = next(iter(loader))
        print("n_batch", len(loader), x.shape, y.shape, type(y))
        TransformDataset.annotate_batch(x, y)
        # annotated_batch = DlibDataModule.draw_batch()
        for x, y in tqdm(datamodule.train_dataloader()):
            pass
        print("training data passed")

        for x, y in tqdm(datamodule.val_dataloader()):
            pass
        print("validation data passed")

        for x, y in tqdm(datamodule.test_dataloader()):
            pass
        print("test data passed")

    # get root dir to this folder
    root = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
    config_path = str(root / "configs" / "data")

    # print("root", root)
    @hydra.main(version_base="1.3", config_path=config_path, config_name="dlib.yaml")
    def main(cfg: DictConfig):
        # print(cfg)
        # print(cfg.batch_size)
        test_dataset(cfg)
        test_datamodule(cfg)
    
    main()

