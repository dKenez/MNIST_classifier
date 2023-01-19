from tests import _PATH_DATA
from src.data.CorruptMNISTDataset import loadCorruptMNIST

def test_data():
    dataset_x, dataset_y = loadCorruptMNIST(_PATH_DATA + "/processed/train.npz")
    assert len(dataset_x) == 40000
    for datapoint_x in dataset_x:
        assert datapoint_x.shape == [1,28,28]
    assert len(dataset_x) == len(dataset_y)
