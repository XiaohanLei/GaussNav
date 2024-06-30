from . import metrics as metrics_module
from .metrics import action_id_to_action_name


def make_metrics(metrics_config):
    return [getattr(metrics_module, metric_name) for metric_name in metrics_config]
