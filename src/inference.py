import torch
import torchvision
import matplotlib.pyplot as plt

from accelerate import Accelerator

from src.model import Classifier
from src.data_processing import get_dataloaders


MODEL_PATH='./model.pt'

if __name__=="__main__":
    """
    Testing out local inference to check model performance.
    """
    model = Classifier()
    model.load_state_dict(torch.load(MODEL_PATH))

    accelerator = Accelerator()

    _, test_loader = get_dataloaders()

    test_loader, model = accelerator.prepare(test_loader, model)
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data

            out = model(images)    

            _, predicted = torch.max(out, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    # prepare to count predictions for each class
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    
