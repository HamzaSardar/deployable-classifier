import torch
import torchvision
import torch.optim as optim
import torch.nn as nn

from accelerate import Accelerator

from src.model import Classifier
from src.data_processing import get_dataloaders


def train(train_dl, model, optim, loss_fn):
    for epoch in range(10):
        loss = 0
        for i, data in enumerate(train_dl):
            x, label = data
            optim.zero_grad()

            out = model(x)
            loss = loss_fn(out, label)

            if i % 2000 == 0:
                print(f'epoch: {epoch + 1}, loss: {loss.item()}')
            
            loss.backward()
            optim.step()

    torch.save(model.state_dict(), 'model.pt')
    print('done')
               

if __name__=="__main__":
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

    train(train_loader, model=model, optim=optimiser, loss_fn=loss)
    
