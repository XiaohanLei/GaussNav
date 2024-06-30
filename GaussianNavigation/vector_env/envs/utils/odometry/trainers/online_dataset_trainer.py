from collections import defaultdict

import numpy as np
import torch
from habitat_baselines.common.tensor_dict import TensorDict
from tqdm import tqdm

from habitat_baselines.rl.ddppo.algo.ddp_utils import rank0_only

from .base_trainer import BaseTrainer
from ..utils import transform_batch
from ..metrics import action_id_to_action_name


class BatchBuffer:
    def __init__(self, max_num_batches, batch_size):
        self.max_num_batches = max_num_batches
        self.num_batches = 0
        self.batch_size = batch_size
        self.size = self.max_num_batches * self.batch_size

        self.buffer = TensorDict()
        self.buffer['source_depth'] = torch.from_numpy(np.zeros((self.size, 1, 180, 320), dtype=np.float32,))
        self.buffer['target_depth'] = torch.from_numpy(np.zeros((self.size, 1, 180, 320), dtype=np.float32,))
        self.buffer['source_rgb'] = torch.from_numpy(np.zeros((self.size, 3, 180, 320), dtype=np.float32,))
        self.buffer['target_rgb'] = torch.from_numpy(np.zeros((self.size, 3, 180, 320), dtype=np.float32,))
        self.buffer['action'] = torch.from_numpy(np.zeros((self.size,), dtype=np.int64,))
        self.buffer['collision'] = torch.from_numpy(np.zeros((self.size,), dtype=np.int64,))

        self.buffer['egomotion'] = TensorDict()
        self.buffer['egomotion']['translation'] = torch.from_numpy(np.zeros((self.size, 3), dtype=np.float32,))
        self.buffer['egomotion']['rotation'] = torch.from_numpy(np.zeros((self.size,), dtype=np.float64,))

    def append(self, batch):
        start, end = self.num_batches * self.batch_size, (self.num_batches + 1) * self.batch_size

        self.buffer['source_depth'][start:end] = batch['source_depth']
        self.buffer['target_depth'][start:end] = batch['target_depth']
        self.buffer['source_rgb'][start:end] = batch['source_rgb']
        self.buffer['target_rgb'][start:end] = batch['target_rgb']
        self.buffer['action'][start:end] = batch['action']
        self.buffer['collision'][start:end] = batch['collision']
        self.buffer['egomotion']['translation'][start:end] = batch['egomotion']['translation']
        self.buffer['egomotion']['rotation'][start:end] = batch['egomotion']['rotation']

        self.num_batches += 1

    def replace(self, batch):
        indices = np.random.choice(self.size - 1, size=self.batch_size, replace=False)

        return_batch = TensorDict()
        return_batch['source_depth'] = self.buffer['source_depth'][indices]
        return_batch['target_depth'] = self.buffer['target_depth'][indices]
        return_batch['source_rgb'] = self.buffer['source_rgb'][indices]
        return_batch['target_rgb'] = self.buffer['target_rgb'][indices]
        return_batch['action'] = self.buffer['action'][indices]
        return_batch['collision'] = self.buffer['collision'][indices]

        return_batch['egomotion'] = TensorDict()
        return_batch['egomotion']['translation'] = self.buffer['egomotion']['translation'][indices]
        return_batch['egomotion']['rotation'] = self.buffer['egomotion']['rotation'][indices]

        self.buffer['source_depth'][indices] = batch['source_depth']
        self.buffer['target_depth'][indices] = batch['target_depth']
        self.buffer['source_rgb'][indices] = batch['source_rgb']
        self.buffer['target_rgb'][indices] = batch['target_rgb']
        self.buffer['action'][indices] = batch['action']
        self.buffer['collision'][indices] = batch['collision']
        self.buffer['egomotion']['translation'][indices] = batch['egomotion']['translation']
        self.buffer['egomotion']['rotation'][indices] = batch['egomotion']['rotation']

        return return_batch

    def __len__(self):
        return self.num_batches


class ShuffleBatchBuffer:
    def __init__(self, dataloader, max_num_batches, batch_size):
        self.dataloader = dataloader
        self.buffer = BatchBuffer(max_num_batches, batch_size)

    def __iter__(self):
        for x in self.dataloader:
            if len(self.buffer) == self.buffer.max_num_batches:
                yield self.buffer.replace(x)
            else:
                self.buffer.append(x)


class OnlineDatasetTrainer(BaseTrainer):
    def init_trainer(self):
        BaseTrainer.init_trainer(self)
        if self.config.train.batch_buffer is not None:
            self.train_loader = ShuffleBatchBuffer(
                self.train_loader,
                max_num_batches=self.config.train.batch_buffer.params.buffer_max_num_batches,
                batch_size=self.config.train.batch_buffer.params.batch_size
            )

    def update_config(self):
        BaseTrainer.update_config(self)
        self.config.defrost()
        self.config.train.dataset.params.seed = self.config.seed
        self.config.freeze()

    def update_distrib_config(self, local_rank):
        self.config.defrost()
        self.config.device = local_rank
        self.config.seed += local_rank * self.config.train.loader.params.num_workers
        self.config.train.dataset.params.seed = self.config.seed
        self.config.train.dataset.params.local_rank = local_rank
        self.config.train.dataset.params.world_size = torch.distributed.get_world_size()
        self.config.train.loader.is_distributed = False
        self.config.val.loader.is_distributed = True
        self.config.freeze()

    def val_epoch(self):
        self.model.eval()

        num_items = 0
        num_items_per_action = defaultdict(lambda: 0)

        metrics = defaultdict(lambda: 0)

        with torch.no_grad():
            for data in tqdm(self.val_loader, disable=self.is_distributed()):
                data, embeddings, target = transform_batch(data)
                data = data.float().to(self.device)
                target = target.float().to(self.device)
                for k, v in embeddings.items():
                    embeddings[k] = v.to(self.device)

                output = self.model(data, **embeddings)
                loss, loss_components = self.loss_f(output, target)

                batch_size = target.shape[0]
                metrics['loss'] += loss.item() * batch_size
                for loss_component, value in loss_components.items():
                    metrics[loss_component] += value.item() * batch_size
                for metric_f in self.val_metric_fns:
                    metrics[metric_f.__name__] += metric_f(output, target).item() * batch_size
                    if self.config.compute_metrics_per_action:
                        for action_id in embeddings['action'].unique():
                            action_name = action_id_to_action_name[action_id.item()]
                            action_mask = embeddings['action'] == action_id
                            action_metric_name = f'{metric_f.__name__}_{action_name}'
                            num_action_items = action_mask.sum()

                            metrics[action_metric_name] += metric_f(output[action_mask], target[action_mask]).item() * num_action_items
                            num_items_per_action[action_metric_name] += num_action_items

                num_items += batch_size

        for metric_name in metrics:
            metrics[metric_name] /= num_items_per_action.get(metric_name, num_items)

        return metrics

    def loop(self):
        self.model.train()

        num_items = 0
        num_items_per_action = defaultdict(lambda: 0)
        train_metrics = defaultdict(lambda: 0)

        for batch_index, data in enumerate(self.train_loader):
            data, embeddings, target = transform_batch(data)
            data = data.float().to(self.device)
            target = target.float().to(self.device)
            for k, v in embeddings.items():
                embeddings[k] = v.to(self.device)

            output = self.model(data, **embeddings)
            loss, loss_components = self.loss_f(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = target.shape[0]
            train_metrics['loss'] += loss.item() * batch_size
            for loss_component, value in loss_components.items():
                train_metrics[loss_component] += value.item() * batch_size
            for metric_f in self.train_metric_fns:
                train_metrics[metric_f.__name__] += metric_f(output, target).item() * batch_size
                if self.config.compute_metrics_per_action:
                    for action_id in embeddings['action'].unique():
                        action_name = action_id_to_action_name[action_id.item()]
                        action_mask = embeddings['action'] == action_id
                        action_metric_name = f'{metric_f.__name__}_{action_name}'
                        num_action_items = action_mask.sum()

                        action_metric_value = metric_f(output[action_mask], target[action_mask]).item()
                        train_metrics[action_metric_name] += action_metric_value * num_action_items
                        num_items_per_action[action_metric_name] += num_action_items

            num_items += batch_size

            if (batch_index + 1) % self.config.batches_per_epoch == 0:
                epoch = (batch_index + 1) // self.config.batches_per_epoch

                for metric_name in train_metrics:
                    train_metrics[metric_name] /= num_items_per_action.get(metric_name, num_items)

                if self.is_distributed():
                    train_metrics = self.coalesce_post_step(train_metrics, self.device)

                # report train metrics:
                self.write_metrics(epoch, train_metrics, self.train_writer)
                self.print_metrics('Train', train_metrics)

                # reset train metrics:
                num_items = 0
                num_items_per_action = defaultdict(lambda: 0)
                train_metrics = defaultdict(lambda: 0)

                # compute val metrics:
                val_metrics = self.val_epoch()
                if self.is_distributed():
                    val_metrics = self.coalesce_post_step(val_metrics, self.device)

                # report val metrics:
                self.write_metrics(epoch, val_metrics, self.val_writer)
                self.print_metrics('Val', val_metrics)

                if rank0_only() and self.config.model.save:
                    best_checkpoint_path = self.config.model.best_checkpoint_path.replace('.pt', f'_{str(epoch).zfill(3)}e.pt')
                    torch.save(self.model.state_dict(), best_checkpoint_path)
                    print('Saved best model checkpoint to disk.')

                if self.scheduler:
                    self.scheduler.step()

                self.model.train()

                if epoch == self.config.epochs:
                    break
