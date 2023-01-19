import argparse
from pathlib import Path
import sys

import torch
import click

from src.data.CorruptMNISTDataset import CorruptMNISTDataset
from src.models.model import MyAwesomeModel
from torch.utils.data import DataLoader
from torch import nn, optim
import matplotlib.pyplot as plt


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option(
    "--checkpoint",
    default="checkpoint",
    help="checkpoint file name without file extension",
)
def main(lr: float, checkpoint: str):
    print("Training day and night")

    root_dir = Path.cwd()

    data_dir = root_dir / "data" / "processed"
    models_dir = root_dir / "models"
    figures_dir = root_dir / "reports" / "figures"

    train_data = CorruptMNISTDataset(data_dir / "train.npz", 64 * 10 * 8)
    test_data = CorruptMNISTDataset(data_dir / "test.npz", 64 * 10 * 2)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)
    model = MyAwesomeModel()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 30

    train_losses = []
    for e in range(epochs):
        running_train_loss = 0
        running_test_loss = 0
        hits = torch.tensor([])
        for images, labels in train_loader:

            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        with torch.no_grad():

            model.eval()
            for images, labels in test_loader:
                log_ps = model(images)

                _, top_class = log_ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                hits = torch.concat([hits, equals])

            accuracy = torch.mean(hits.type(torch.FloatTensor))

        train_losses.append(running_train_loss)

        model.train()
        click.echo(
            f"epoch: {e}\tTrain loss: {running_train_loss:.2f}\tAccuracy: {accuracy.item():.2%}"
        )

    torch.save(model.state_dict(), models_dir / f"{checkpoint}.pth")

    epoch_list = list(range(epochs))
    plt.plot(epoch_list, train_losses)
    # plt.plot(epoch_list, test_losses)
    plt.savefig(figures_dir / f"{checkpoint}.png")

    click.echo(f"Saved model to: {checkpoint}.pth")
    click.echo(f"Saved training data t: {checkpoint}.png")


if __name__ == "__main__":
    main()
