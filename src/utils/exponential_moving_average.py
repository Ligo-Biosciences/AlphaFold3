from collections import OrderedDict
import copy
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel
from src.utils.tensor_utils import tensor_tree_map


class ExponentialMovingAverage(AveragedModel):
    """
    Maintains moving averages of parameters with exponential decay

    At each step, the stored copy `copy` of each parameter `param` is
    updated as follows:

        `copy = decay * copy + (1 - decay) * param`

    where `decay` is an attribute of the ExponentialMovingAverage object.
    """

    def __init__(self, model: nn.Module, decay: float):
        """
        Args:
            model:
                A torch.nn.Module whose parameters are to be tracked
            decay:
                A value (usually close to 1.) by which updates are
                weighted as part of the above formula
        """
        super(ExponentialMovingAverage, self).__init__(
            model=model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay)
        )
        self.decay = decay
