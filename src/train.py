import os
import json
import argparse
import sys
import warnings
from pathlib import Path
from ast import literal_eval
warnings.filterwarnings('ignore')

import torch
import torchvision as tv
import pytorch_lightning as pl
import webdataset as wds
from sm_resnet.models import ResNet
from sm_resnet.callbacks import PlSageMakerLogger, ProfilerCallback, SMDebugCallback
from sm_resnet.utils import get_training_world, get_rank
import smdebug.pytorch as smd
from smdebug.core.reduction_config import ReductionConfig
from smdebug.core.save_config import SaveConfig
from smdebug.core.collection import CollectionKeys
# from smdebug.core.utils import check_sm_training_env

world = get_training_world()

def parse_args():
    cmdline = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdline.add_argument('--train_file_dir', default='/opt/ml/input/data/train/',
                         help="""Path to dataset in WebDataset format.""")
    cmdline.add_argument('--validation_file_dir', default='/opt/ml/input/data/val/',
                         help="""Path to dataset in WebDataset format.""")
    cmdline.add_argument('--max_epochs', default=20, type=int,
                         help="""Number of epochs.""")
    cmdline.add_argument('--num_classes', default=1000, type=int,
                         help="""Number of classes.""")
    cmdline.add_argument('--resnet_version', default=50, type=int,
                         help="""Resnet version.""")
    cmdline.add_argument('-lr', '--learning_rate', default=1e-2, type=float,
                         help="""Base learning rate.""")
    cmdline.add_argument('-b', '--batch_size', default=128, type=int,
                         help="""Size of each minibatch per GPU""")
    cmdline.add_argument('--warmup_epochs', default=1, type=int,
                         help="""Number of epochs for learning rate warmup""")
    cmdline.add_argument('--mixup_alpha', default=0.0, type=float,
                         help="""Extent of convex combination for training mixup""")
    cmdline.add_argument('--optimizer', default='adamw', type=str,
                         help="""Optimizer type""")
    cmdline.add_argument('--strategy', default='horovod', type=str,
                         help="""Distribution strategy""")
    cmdline.add_argument('--precision', default=16, type=int,
                         help="""Floating point precision""")
    cmdline.add_argument('--profiler_start', default=128, type=int,
                         help="""Profiler start step""")
    cmdline.add_argument('--profiler_steps', default=32, type=int,
                         help="""Profiler steps""")
    cmdline.add_argument('--dataloader_workers', default=4, type=int,
                         help="""Number of data loaders""")
    cmdline.add_argument('--debugging_output', default='/opt/ml/debugger/',
                         help="""Path to dataset in WebDataset format.""")
    cmdline.add_argument('--train_batches', default=1, type=int,
                         help="""Number of batches to use for each training epoch""")
    return cmdline
    
def main(ARGS):
    
    model_params = {'num_classes': ARGS.num_classes,
                    'resnet_version': ARGS.resnet_version,
                    'train_path': ARGS.train_file_dir,
                    'val_path': ARGS.validation_file_dir,
                    'optimizer': ARGS.optimizer,
                    'lr': ARGS.learning_rate, 
                    'batch_size': ARGS.batch_size,
                    'dataloader_workers': ARGS.dataloader_workers,
                    'max_epochs': ARGS.max_epochs,
                    'warmup_epochs': ARGS.warmup_epochs,
                    'mixup_alpha': ARGS.mixup_alpha
                   }

    trainer_params = {'strategy': ARGS.strategy,
                      'gpus': world["size"],
                      'num_nodes': world["number_of_machines"],
                      'max_epochs': ARGS.max_epochs,
                      'precision': ARGS.precision,
                      'limit_train_batches': float(ARGS.train_batches) if ARGS.train_batches==1.0 else int(ARGS.train_batches),
                      'progress_bar_refresh_rate': 0,
                      'callbacks': [PlSageMakerLogger(),
                                    SMDebugCallback()]
                      }
    
    model = ResNet(**model_params)
    trainer = pl.Trainer(**trainer_params)
    
    '''
    # Setup Debugger
    if check_sm_training_env():
        smd.Hook.register_hook(model, model.criterion)
    else:
        reduction_config = ReductionConfig(['mean'])
        save_config = SaveConfig(save_interval=25)
        include_collections = [CollectionKeys.LOSSES]
        hook = smd.Hook(out_dir='./smdebugger',
                        export_tensorboard=True,
                        tensorboard_dir='./tensorboard',
                        reduction_config=reduction_config,
                        save_config=save_config,
                        include_regex=None,
                        include_collections=include_collections,
                        save_all=False,)
    '''
    
    trainer.fit(model)

if __name__=='__main__':
    cmdline = parse_args()
    ARGS, unknown_args = cmdline.parse_known_args()
    main(ARGS)
        