"""MLflow-enabled trainng and eval pipeline."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
import mlflow
from mlflow.tracking import MlflowClient
from absl import app, flags
from ml_collections import config_flags

from src.model import Classifier
from src.data_processing import get_dataloaders
from src.train import train
from src.inference import evaluate

_CONFIG = config_flags.DEFINE_config_file(
    "config",
    str(Path(__file__).resolve().parent / "configs" / "config.py"),
    "Path to the config file.",
)

_DEFAULT_EXPERIMENT_NAME = "cifar-10-classifier"

flags.DEFINE_boolean("train", False, "Train the model.")
flags.DEFINE_boolean("eval", False, "Evaluate the trained model.")
flags.DEFINE_boolean("allrun", False, "Run train and eval in sequence.")
flags.DEFINE_string("experiment", _DEFAULT_EXPERIMENT_NAME, "MLflow experiment name.")
flags.DEFINE_string("run_id", None, "Run ID if evaluating existing model.")
flags.register_multi_flags_validator(
    ["eval", "run_id"],
    lambda f: not f["eval"] or f["run_id"] is not None,
    message="--run_id is required when --eval is set",
)


def main(_: list[str]) -> None:
    """Run pipeline steps according to the provided flags.

    Args:
        _: Remaining argv after abseil flag parsing. Unused but required by abseil.
    """
    absl_flags = flags.FLAGS
    absl_cfg = absl_flags.config

    mlflow.set_experiment(absl_flags.experiment)

    train_loader, test_loader = get_dataloaders(batch_size=absl_cfg.training.batch_size)
    model = Classifier()
    accelerator = Accelerator()

    with mlflow.start_run():
        if absl_flags.allrun or absl_flags.train:
            # initialise loss and optimiser
            loss = nn.CrossEntropyLoss()
            optimiser = optim.Adam(params=model.parameters(), lr=absl_cfg.training.lr)

            # pass everything to GPU if available
            train_loader, test_loader, model, optimiser = accelerator.prepare(
                train_loader, test_loader, model, optimiser
            )

            mlflow.log_params(
                {
                    "batch_size": absl_cfg.training.batch_size,
                    "n_epochs": absl_cfg.training.n_epochs,
                    "lr": absl_cfg.training.lr,
                }
            )

            active_run_id = mlflow.active_run()
            if active_run_id:
                active_run_id = active_run_id.info.run_id

            model = train(
                train_loader,
                model=model,
                optim=optimiser,
                loss_fn=loss,
                n_epochs=absl_cfg.training.n_epochs,
                run_id=active_run_id,
            )

            model_uri = f"runs:/{active_run_id}/model"
            mlflow.register_model(model_uri=model_uri, name="cifar10-classifier")

            client = MlflowClient()

            latest_version = client.get_latest_versions(
                name="cifar10-classifier", stages=["none"]
            )[0].version
            client.transition_model_version_stage(
                name="cifar10-classifier", version=latest_version, stage="Production"
            )

        if absl_flags.eval or absl_flags.allrun:
            if absl_flags.eval:
                model = mlflow.pytorch.load_model(f"runs:/{absl_flags.run_id}/model")
                _, test_loader = get_dataloaders()
                model, test_loader = accelerator.prepare(model, test_loader)

            results_dict = evaluate(model, accelerator, test_loader, absl_cfg.eval)
            mlflow.log_metrics(results_dict)


if __name__ == "__main__":
    app.run(main)
