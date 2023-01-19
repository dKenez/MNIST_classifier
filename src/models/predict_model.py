from pathlib import Path


import torch
import click

from src.data.CorruptMNISTDataset import CorruptMNISTDataset
from src.models.model import MyAwesomeModel

import helper


@click.command()
@click.argument("i")
@click.option("--checkpoint", default="checkpoint", help="Checkpoint file")
def main(i, checkpoint):

    print("Evaluating until hitting the ceiling")

    root_dir = Path.cwd()

    data_dir = root_dir / "data" / "processed"
    models_dir = root_dir / "models"

    state_dict = torch.load(models_dir / f"{checkpoint}.pth")

    model = MyAwesomeModel()
    model.load_state_dict(state_dict)

    model.eval()

    test_data = CorruptMNISTDataset(data_dir / "test.npz")

    img, _ = test_data[int(i)]
    # Convert 2D image to 1D vector
    img = img.view(1, 784)

    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(img)

    ps = torch.exp(output)
    _, top_class = output.topk(1, dim=1)
    click.echo(int(top_class[0,0]))
    # Plot the image and probabilities
    # helper.view_classify(img.view(1, 28, 28), ps)


if __name__ == "__main__":
    main()
