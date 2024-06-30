from . import schedulers as schedulers_module


def make_scheduler(scheduler_config, optimizer):
    scheduler_type = getattr(schedulers_module, scheduler_config.type)
    scheduler = scheduler_type(
        optimizer,
        **(scheduler_config.params if scheduler_config.params is not None else {})
    )

    return scheduler
