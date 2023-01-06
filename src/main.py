import argparse
from pathlib import Path
import sys

import torch
import click

from data import CorruptMNISTDataset
from model import MyAwesomeModel
from torch.utils.data import DataLoader
from torch import nn, optim

import helper

@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
@click.option("--checkpoint", default="model.pth", help='Chackpoint file')
def train(lr, checkpoint):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    data_folder = Path(__file__).parent.parent / "corruptmnist"

    train_data = CorruptMNISTDataset(data_folder / "train_0.npz", 800)
    test_data = CorruptMNISTDataset(data_folder / "train_1.npz", 200)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)
    model = MyAwesomeModel()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 15

    train_losses, test_losses = [], []
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_loader:

            optimizer.zero_grad()
            
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        else:
            ## TODO: Implement the validation pass and print out the validation accuracy
            with torch.no_grad():

                model.eval()
                for images, labels in test_loader:
                    ps = model(images)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy = torch.mean(equals.type(torch.FloatTensor))

            model.train()
            click.echo(f'Accuracy: {accuracy.item()*100}%')

    torch.save(model.state_dict(), checkpoint)
    click.echo(f'Saved model tp: {checkpoint}')


@click.command()
@click.argument("i")
@click.option("--checkpoint", default="model.pth", help='Chackpoint file')
def evaluate(i, checkpoint):

    print("Evaluating until hitting the ceiling")

    # TODO: Implement evaluation logic here
    state_dict = torch.load(checkpoint)

    model = MyAwesomeModel()
    model.load_state_dict(state_dict)

    model.eval()

    data_folder = Path(__file__).parent.parent / "corruptmnist"
    test_data = CorruptMNISTDataset(data_folder / "train_2.npz")

    img, _ = test_data[int(i)]
    # Convert 2D image to 1D vector
    img = img.view(1, 784)

    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(img)

    ps = torch.exp(output)

    # Plot the image and probabilities
    helper.view_classify(img.view(1, 28, 28), ps)


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()

  