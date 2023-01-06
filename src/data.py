import numpy as np

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CorruptMNISTDataset(Dataset):
    def __init__(self, npz_file_path, size=None):
        self.images, self.labels = loadCorruptMNIST(npz_file_path, size)
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = self.transform(image)

        label = self.labels[idx]
        return image, label


def loadCorruptMNIST(path, size=None):

    with np.load(path) as data_file:
        sli = slice(size) or ...
        data_x = data_file["images"][sli].astype("float32")
        data_y = data_file["labels"][sli]

    return data_x, data_y


if __name__ == "__main__":
    data_folder = Path(__file__).parent.parent / "corruptmnist"

    test_x, test_y = loadCorruptMNIST(data_folder / "train_0.npz", 800)
    print(len(test_x))
    test_x, test_y = loadCorruptMNIST(data_folder / "train_0.npz")
    print(len(test_x))
    test_x, test_y = loadCorruptMNIST(data_folder / "train_0.npz", -5)
    print(len(test_x))
    # test_x, test_y = loadCorruptMNIST(data_folder / "train_0.npz", 0)
    # print(len(test_x))

    corruptMNISTDataset = CorruptMNISTDataset(data_folder / "train_0.npz")
    print(f"dataset length: {len(corruptMNISTDataset)}")

    test_image, test_label = corruptMNISTDataset[0]
    print(f"test image shape: {test_image.shape}")
    print(f"test label: {test_label}")
