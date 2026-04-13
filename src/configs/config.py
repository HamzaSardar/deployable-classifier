import ml_collections

from src.configs import common


def get_config() -> ml_collections.ConfigDict:

    config = common.get_config()

    config.training = ml_collections.ConfigDict()
    config.training.n_epochs = 10
    config.training.batch_size = 32
    config.training.lr = 0.001

    config.eval = ml_collections.ConfigDict()
    config.eval.classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    return config
