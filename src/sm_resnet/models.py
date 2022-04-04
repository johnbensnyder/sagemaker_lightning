import os
import torch
import torchvision as tv
from torch.cuda.amp import autocast, GradScaler
import pytorch_lightning as pl
import pl_bolts
from sm_resnet.utils import is_smddp
from torch.nn.parallel import DistributedDataParallel as DDP
from sm_resnet.data import mixup_data, mixup_criterion, build_dataloader

from torch.cuda.amp import autocast, GradScaler

world_size = int(os.environ.get("WORLD_SIZE", 1))
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))

class ResNet(pl.LightningModule):
    
    def __init__(self, num_classes, 
                 resnet_version,
                 train_path=None, 
                 val_path=None, 
                 optimizer='adamw',
                 lr=1e-3, 
                 batch_size=64,
                 dataloader_workers=4, 
                 max_epochs=20,
                 warmup_epochs=1,
                 mixup_alpha=0.,
                 *args, **kwargs):
        super().__init__()
        self.automatic_optimization = False
        
        self.__dict__.update(locals())
        
        resnets = {
            18:tv.models.resnet18,
            34:tv.models.resnet34,
            50:tv.models.resnet50,
            101:tv.models.resnet101,
            152:tv.models.resnet152
        }
        
        optimizers = {
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'sgd': torch.optim.SGD
        }
        
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.model = resnets[resnet_version]()
        linear_size = list(self.model.children())[-1].in_features
        self.model.fc = torch.nn.Linear(linear_size, num_classes)
        '''if world_size>1:
            device = torch.device(f'cuda:{local_rank}') 
            self.model = DDP(self.model.to(device), device_ids=[local_rank])'''
            
        self.optimizer = optimizers[optimizer]
        
    def configure_optimizers(self):
        opt = self.optimizer(self.parameters(), lr=self.lr)
        self.schedule = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(opt,
                                                                                       self.warmup_epochs,
                                                                                       self.max_epochs)
        return opt
    
    def forward(self, X):
        return self(X)
    
    def train_dataloader(self):
        return build_dataloader(self.train_path, self.batch_size, self.dataloader_workers, train=True)
    
    def val_dataloader(self):
        return build_dataloader(self.val_path, self.batch_size, self.dataloader_workers, train=False)
    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        self.x, self.y, self.idx = batch
        if self.mixup_alpha>0.:
            mixed_x, y_a, y_b, lam = mixup_data(self.x, self.y, self.mixup_alpha)
            self.preds = self(mixed_x)
            loss = mixup_criterion(self.criterion, self.preds, y_a, y_b, lam)
        else:
            self.preds = self(self.x)
            loss = self.criterion(self.preds, self.y)
        self.manual_backward(loss)
        opt.step()
        self.schedule.step()
        acc = (self.y == torch.argmax(self.preds, 1)).type(torch.FloatTensor).mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=False, sync_dist=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=False, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, idx = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        acc = (y == torch.argmax(preds, 1)).type(torch.FloatTensor).mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=False, sync_dist=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=False, sync_dist=True)
        return acc 
