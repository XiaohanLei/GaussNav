import os
import shutil
import random

import numpy as np
import torch
import torch.distributed
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

from habitat_baselines.rl.ddppo.algo.ddp_utils import get_distrib_size, init_distrib_slurm, rank0_only

from ..dataset import make_dataset, make_data_loader
from ..models import make_model
from ..metrics import make_metrics
from ..optims import make_optimizer
from ..losses import make_loss
from ..models.models import init_distributed
from ..schedulers import make_scheduler


def convert_to_refactored_vo_state_dict(checkpoint):
    new_checkpoint = OrderedDict()
    for k, v in checkpoint.items():
        if k.startswith('encoder.0.'):
            new_checkpoint[k.replace('encoder.0.', 'encoder.')] = v
        elif k == 'encoder.1.weight':
            new_checkpoint['compression.0.weight'] = v
        elif k == 'encoder.2.weight':
            new_checkpoint['compression.1.weight'] = v
        elif k == 'encoder.2.bias':
            new_checkpoint['compression.1.bias'] = v
        else:
            new_checkpoint[k] = v

    return new_checkpoint


class BaseTrainer:
    def __init__(self, config):
        self.config = config

        self.train_dataset = None
        self.train_loader = None
        self.train_metric_fns = None

        self.val_dataset = None
        self.val_loader = None
        self.val_metric_fns = None

        self.device = None
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.warmup_scheduler = None
        self.loss_f = None

        self.train_writer = None
        self.val_writer = None

    def init_trainer(self):
        self.train_dataset = make_dataset(self.config.train.dataset)
        self.train_loader = make_data_loader(self.config.train.loader, self.train_dataset)
        self.train_metric_fns = make_metrics(self.config.train.metrics) if self.config.train.metrics else []

        self.val_dataset = make_dataset(self.config.val.dataset)
        self.val_loader = make_data_loader(self.config.val.loader, self.val_dataset)
        self.val_metric_fns = make_metrics(self.config.val.metrics) if self.config.val.metrics else []

        self.device = torch.device(self.config.device)
        self.model = make_model(self.config.model).to(self.device)
        if hasattr(self.config.model, 'pretrained_checkpoint') and self.config.model.pretrained_checkpoint is not None:
            checkpoint = torch.load(self.config.model.pretrained_checkpoint, map_location=self.device)
            # remove 'module.' from state dict key if model was trained in distributed mode
            new_checkpoint = OrderedDict()
            for k, v in checkpoint.items():
                new_checkpoint[k.replace('module.', '')] = v
            checkpoint = new_checkpoint
            try:
                self.model.load_state_dict(checkpoint)
            except RuntimeError as e:
                checkpoint = convert_to_refactored_vo_state_dict(checkpoint)
                self.model.load_state_dict(checkpoint)

        if self.is_distributed():
            self.model = init_distributed(self.model, self.device, find_unused_params=True)

        self.loss_f = make_loss(self.config.loss)
        self.optimizer = make_optimizer(self.config.optim, self.model.parameters())
        self.warmup_scheduler = (
            make_scheduler(self.config.schedulers.warmup, self.optimizer)
            if hasattr(self.config, 'schedulers') and self.config.schedulers.warmup is not None else None
        )
        self.lr_scheduler =(
            make_scheduler(self.config.schedulers.lr, self.optimizer)
            if hasattr(self.config, 'schedulers') and self.config.schedulers.lr is not None else None
        )

    def init_distrib(self):
        local_rank, tcp_store = init_distrib_slurm(self.config.distrib_backend)
        if rank0_only():
            print("Initialized with {} workers".format(torch.distributed.get_world_size()))
        self.update_distrib_config(local_rank)

    def update_config(self):
        self.config.defrost()
        self.config.experiment_dir = os.path.join(self.config.log_dir, self.config.experiment_name)
        self.config.tb_dir = os.path.join(self.config.experiment_dir, 'tb')
        self.config.model.best_checkpoint_path = os.path.join(self.config.experiment_dir, 'best_checkpoint.pt')
        self.config.model.last_checkpoint_path = os.path.join(self.config.experiment_dir, 'last_checkpoint.pt')
        self.config.config_save_path = os.path.join(self.config.experiment_dir, 'config.yaml')
        self.config.freeze()

    def update_distrib_config(self, local_rank):
        raise NotImplemented

    def init_writers(self):
        self.train_writer = SummaryWriter(log_dir=os.path.join(self.config.tb_dir, 'train'))
        self.val_writer = SummaryWriter(log_dir=os.path.join(self.config.tb_dir, 'val'))

    def close_writers(self):
        self.train_writer.close()
        self.val_writer.close()

    @rank0_only
    def init_experiment(self):
        if os.path.exists(self.config.experiment_dir):
            def ask():
                return input(f'Experiment "{self.config.experiment_dir}" already exists. Delete (y/n)?')

            answer = ask()
            while answer not in ('y', 'n'):
                answer = ask()

            delete = answer == 'y'
            if not delete:
                exit(1)

            shutil.rmtree(self.config.experiment_dir)

        os.makedirs(self.config.experiment_dir)
        with open(self.config.config_save_path, 'w') as dest_file:
            self.config.dump(stream=dest_file)

    def set_random_seed(self):
        seed = self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def loop(self):
        raise NotImplemented

    def train(self):
        self.update_config()
        if self.is_distributed():
            self.init_distrib()

        self.init_experiment()
        self.set_random_seed()

        self.init_trainer()

        self.init_writers()
        self.loop()
        self.close_writers()

    @staticmethod
    def is_distributed():
        return get_distrib_size()[2] > 1

    @staticmethod
    def _all_reduce(t: torch.Tensor, device) -> torch.Tensor:
        orig_device = t.device
        t = t.to(device)
        torch.distributed.all_reduce(t)

        return t.to(orig_device)

    @staticmethod
    def coalesce_post_step(metrics, device):
        metric_name_ordering = sorted(metrics.keys())
        stats = torch.tensor(
            [metrics[k] for k in metric_name_ordering],
            device="cpu",
            dtype=torch.float32,
        )
        stats = BaseTrainer._all_reduce(stats, device)
        stats /= torch.distributed.get_world_size()

        return {
            k: stats[i].item() for i, k in enumerate(metric_name_ordering)
        }

    @staticmethod
    @rank0_only
    def print_metrics(phase, metrics):
        metrics_log_str = ' '.join([
            '\t{}: {:.6f}\n'.format(k, v)
            for k, v in metrics.items()
        ])

        print(f'{phase}:\n {metrics_log_str}')

    @staticmethod
    @rank0_only
    def write_metrics(epoch, metrics, writer):
        for metric_name, value in metrics.items():
            key = 'losses' if 'loss' in metric_name else 'metrics'
            writer.add_scalar(f'{key}/{metric_name}', value, epoch)
