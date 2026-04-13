from pathlib import Path

import ml_collections
import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from accelerate import Accelerator

from src.model import Classifier
from src.data_processing import get_dataloaders


MODEL_PATH = Path("./model.pt")


def evaluate(
    model: Classifier,
    accelerator: Accelerator,
    test_loader: DataLoader,
    eval_config: ml_collections.ConfigDict,
) -> dict[str, float]:

    correct = 0
    total = 0
    eval_dict = {}

    with torch.no_grad():
        for data in test_loader:
            images, labels = data

            out = model(images)

            _, predicted = torch.max(out, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"Accuracy of the network on the 10000 test images: {100 * correct / total} %"
    )
    eval_dict["overall"] = 100 * correct / total

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in eval_config.classes}
    total_pred = {classname: 0 for classname in eval_config.classes}

    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[eval_config.classes[label.item()]] += 1
                total_pred[eval_config.classes[label.item()]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")
        eval_dict[classname] = accuracy

    return eval_dict


if __name__ == "__main__":
    """
    Testing out local inference to check model performance.
    """
    from src.configs.config import get_config

    cfg = get_config()

    model = Classifier()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    accelerator = Accelerator()
    _, test_loader = get_dataloaders()

    results = evaluate(model, accelerator, test_loader, cfg.eval)
    print(results)
