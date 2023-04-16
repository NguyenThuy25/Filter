from pathlib import Path

import pytest
import torch

from src.data.dlib_datamodule import DlibDataModule


@pytest.mark.parametrize("batch_size", [32])
def test_dlib_datamodule(batch_size):
    data_dir = "data/"

    dm = DlibDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "Dlib").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()
    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 3837
    # train = dm.train_dataloader()
    batch = next(iter(dm.train_dataloader()))
    # x, y = batch
    # assert len(x) == batch_size
    # assert len(y) == batch_size
    # assert x.dtype == torch.float32
    # assert y.dtype == torch.int64
