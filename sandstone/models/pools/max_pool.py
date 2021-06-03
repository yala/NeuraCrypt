import torch
import torch.nn as nn
from sandstone.models.pools.abstract_pool import AbstractPool
from sandstone.models.pools.factory import RegisterPool

@RegisterPool('GlobalMaxPool')
class GlobalMaxPool(AbstractPool):

    def replaces_fc(self):
        return False

    def forward(self, x, batch):
        spatially_flat_size = (*x.size()[:2], -1)
        x = x.view(spatially_flat_size)
        x, _ = torch.max(x, dim=-1)
        return None, x

@RegisterPool('PerFrameMaxPool')
class PerFrameMaxPool(AbstractPool):
    def __init__(self, args, num_chan):
        super(PerFrameMaxPool, self).__init__(args, num_chan)
        self.globalmaxpool = GlobalMaxPool(args, num_chan)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=args.dropout)
        self.fc = nn.Linear(num_chan, args.num_classes)
        
    def replaces_fc(self):
        return True

    def forward(self, x, batch):
        assert len(x.shape) == 5
        _, hidden = self.globalmaxpool(x, batch)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        logit = self.fc(hidden)
        spatially_flat_size = (*x.size()[:3], -1)
        x = x.view(spatially_flat_size)
        x, _ = torch.max(x, dim=-1)
        return logit, x