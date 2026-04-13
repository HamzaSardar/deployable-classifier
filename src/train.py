"""Training script for classification CNN."""

from typing import Callable

import torch
import torch.optim as optim
from torch.optim import Optimizer
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow

from accelerate import Accelerator

from src.model import Classifier
from src.data_processing import get_dataloaders


def train(
    train_dl: DataLoader,
    model: Callable,
    optim: Optimizer,
    loss_fn: Callable,
    n_epochs: int = 10,
    model_path: str = "model.pt",
    run_id: str | None = None,
):
    """
    Simple training loop for classification CNN.
    """
    step = 0
    for epoch in range(n_epochs):
        loss = 0
        for i, data in enumerate(train_dl):
            x, label = data
            optim.zero_grad()

            out = model(x)
            loss = loss_fn(out, label)

            if i % 2000 == 0:
                print(f"epoch: {epoch + 1}, loss: {loss.item()}")
                if run_id:
                    mlflow.log_metric(
                        "loss",
                        loss.item(),
                        step=step,
                        run_id=run_id,
                    )

            loss.backward()
            optim.step()
            step += 1

    torch.save(model.state_dict(), model_path)
    if run_id:
        mlflow.pytorch.log_model(model, "model")
    return model


if __name__ == "__main__":
    # initialise dataloaders and model
    train_loader, test_loader = get_dataloaders()
    model = Classifier()

    # initialise loss and optimiser
    loss = nn.CrossEntropyLoss()
    optimiser = optim.Adam(params=model.parameters(), lr=0.001)

    # pass everything to GPU
    accelerator = Accelerator()
    train_loader, test_loader, model, optimiser = accelerator.prepare(
        train_loader, test_loader, model, optimiser
    )

    train(
        train_loader,
        model=model,
        optim=optimiser,
        loss_fn=loss,
    )
