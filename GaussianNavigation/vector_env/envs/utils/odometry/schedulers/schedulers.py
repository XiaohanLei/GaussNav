# from pytorch_warmup import UntunedLinearWarmup, UntunedExponentialWarmup # noqa
from .warmup_schedulers import UntunedLinearWarmup # noqa

# lr schedulers:
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR # noqa
