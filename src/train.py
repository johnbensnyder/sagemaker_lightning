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
from sm_resnet.callbacks import PlSageMakerLogger, ProfilerCallback
from sm_resnet.utils import is_sm, is_smddp, get_training_world
if is_smddp():
    import smdistributed.dataparallel.torch.torch_smddp
import torch.distributed as dist

import smdebug.pytorch as smd

world_size = int(os.environ.get("WORLD_SIZE", 1))
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world = get_training_world()

if world_size>1:
    dist.init_process_group(
            backend="smddp" if is_smddp() else "nccl", init_method="env://",
        )

def parse_args():
    cmdline = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdline.add_argument('--train_file_dir', default='/opt/ml/input/data/train/',
                         help="""Path to dataset in WebDataset format.""")
    cmdline.add_argument('--validation_file_dir', default='/opt/ml/input/data/validation/',
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
    cmdline.add_argument('--mixup_alpha', default=0.1, type=float,
                         help="""Extent of convex combination for training mixup""")
    cmdline.add_argument('--optimizer', default='adamw', type=str,
                         help="""Optimizer type""")
    cmdline.add_argument('--precision', default=16, type=int,
                         help="""Floating point precision""")
    cmdline.add_argument('--profiler_start', default=128, type=int,
                         help="""Profiler start step""")
    cmdline.add_argument('--profiler_steps', default=32, type=int,
                         help="""Profiler steps""")
    cmdline.add_argument('--dataloader_workers', default=4, type=int,
                         help="""Number of data loaders""")
    cmdline.add_argument('--logging_output', default='/opt/ml/checkpoints/logging/',
                         help="""Path to dataset in WebDataset format.""")
    cmdline.add_argument('--debugging_output', default='/opt/ml/checkpoints/debugger/',
                         help="""Path to dataset in WebDataset format.""")
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

    trainer_params = {'accelerator': 'gpu',
                      'max_epochs': ARGS.max_epochs,
                      'precision': ARGS.precision,
                      'progress_bar_refresh_rate': 0,
                      'replace_sampler_ddp': True,
                      'num_nodes': world['number_of_machines'],
                      'devices': world['number_of_processes'],
                      'strategy': 'ddp',
                      'callbacks': [PlSageMakerLogger()]
                      }
    
    if local_rank==0:
        trainer_params['callbacks'].append(ProfilerCallback(output_dir=(os.path.join(ARGS.debugging_output, 'profiling'))))
    
    torch.cuda.set_device(local_rank)
    model = ResNet(**model_params)
    trainer = pl.Trainer(**trainer_params)
    
    if is_sm():
        try:
            debugger_hook = smd.Hook.create_from_json_file()
            debugger_hook.register_module(model)
            debugger_hook.register_loss(model.criterion)
        except:
            pass
    else:
        debugger_hook = smd.Hook(out_dir=ARGS.debugging_output, 
                                 include_collections=[],
                                 save_config=SaveConfig(save_interval=10),
                                 export_tensorboard=True,
                                 tensorboard_dir=os.path.join(ARGS.debugging_output, 'tensorboard'))
    
        debugger_hook.register_module(model)
        debugger_hook.register_loss(model.criterion)

    trainer.fit(model)

if __name__=='__main__':
    cmdline = parse_args()
    ARGS, unknown_args = cmdline.parse_known_args()
    main(ARGS)
        