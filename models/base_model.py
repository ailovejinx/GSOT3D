import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, cfg, log):
        super().__init__()
        self.cfg = cfg
        self.log = log
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            'forward has not been implemented!')

