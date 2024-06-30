from . import optims as optims_module


def make_optimizer(optimizer_config, model_parameters):
    optimizer_type = getattr(optims_module, optimizer_config.type)
    optimizer = optimizer_type(
        model_parameters,
        **optimizer_config.params
    )

    return optimizer
