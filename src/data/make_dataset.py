# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np


def read_npz(raw_data_path):
    """
    Load data from .npz file.

    This function reads a .npz file and returns its contents as a tuple containing numpy arrays.

    Args:
    raw_data_path (str): The path to the .npz file.

    Returns:
    tuple: A tuple containing the images and labels arrays. The images array has shape
        (num_images, image_height, image_width), and the labels array has shape (num_labels,).
    """
    with np.load(raw_data_path) as raw_data_file:
        images = raw_data_file["images"].astype("float32")
        labels = raw_data_file["labels"]

    return images, labels


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: str, output_filepath: str):
    """
    Process raw data and save it as processed data.

    This function reads raw data from the input directory, processes it, and saves the processed data as
    .npz files in the output directory. The raw data is assumed to be .npz files containing images and labels.
    The processed data is split into train and test sets, with the train set containing 80% of the data andaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    the test set containing the remaining 20%.

    Args:
    input_filepath (str): The path to the directory containing the raw data.
    output_filepath (str): The path to the directory where the processed data should be saved.

    Returns:
    None
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    root_dir = Path.cwd()

    raw_dir = root_dir / input_filepath
    processed_dir = root_dir / output_filepath

    raw_data_paths = list(raw_dir.glob("*.npz"))
    raw_data_paths.sort()

    # extract images and labels from all .npz files in the input dir
    data_xy_list = [read_npz(raw_data_path) for raw_data_path in raw_data_paths]

    # separate images and labels into data_x and data_y
    data_x = np.concatenate([data_xy[0] for data_xy in data_xy_list], axis=0)
    data_y = np.concatenate([data_xy[1] for data_xy in data_xy_list], axis=0)

    data_len = data_x.shape[0]

    # calc mean and std, then reshape them to be compatible for subtraction and dividion with data_x
    mean_x = np.mean(data_x, axis=(1, 2)).reshape((data_len, 1, 1))
    std_x = np.std(data_x, axis=(1, 2)).reshape((data_len, 1, 1))

    data_x -= mean_x
    data_x /= std_x

    train_x = data_x[: int(data_len * 0.8)]
    train_y = data_y[: int(data_len * 0.8)]
    test_x = data_x[int(data_len * 0.8) :]
    test_y = data_y[int(data_len * 0.8) :]

    np.savez(processed_dir / "train.npz", images=train_x, labels=train_y)
    np.savez(processed_dir / "test.npz", images=test_x, labels=test_y)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
