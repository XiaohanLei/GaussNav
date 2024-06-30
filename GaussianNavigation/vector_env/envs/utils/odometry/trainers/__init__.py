from .static_dataset_trainer import StaticDatasetTrainer
from .online_dataset_trainer import OnlineDatasetTrainer


def make_trainer(config):
    trainer_type = globals()[config.trainer.type]
    trainer = trainer_type(config)

    return trainer
