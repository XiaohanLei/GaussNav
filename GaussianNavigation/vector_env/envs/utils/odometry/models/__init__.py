from . import models as models_module


def make_model(model_config):
    model_type = getattr(models_module, model_config.type)
    model = model_type.from_config(
        model_config
    )

    return model
